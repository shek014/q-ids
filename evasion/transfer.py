"""Offline scoring of a capture-once sweep (evasion/sweep.py) against BOTH detectors. Because every
flow is scored by both models, this gives (a) a PAIRED minimal-jitter comparison on identical traffic
and (b) transferability: does one model's evading beacon also fool the other? No captures here, so
it runs in the .venv without root:

    .venv/bin/python -m evasion.transfer --sweep results/evasion/sweep.npz
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from evasion.detector import Detector


def min_evading_jitter(detector, sweep, threshold):
    """Per seed: smallest jitter (ascending) at which the detector calls the flow benign AND it is
    functional. Returns (evaded[bool], min_jitter[float], min_idx[int])."""
    features, functional, got, jitters = (sweep["features"], sweep["functional"],
                                          sweep["got"], sweep["jitters"])
    n_s, n_j, _ = features.shape
    evaded = np.zeros(n_s, bool)
    min_jitter = np.full(n_s, np.inf)
    min_idx = np.full(n_s, -1, int)
    for si in range(n_s):
        for ji in range(n_j):
            if not got[si, ji]:
                continue
            if functional[si, ji] and detector.prob_benign(features[si, ji])[0] >= threshold:
                evaded[si], min_jitter[si], min_idx[si] = True, float(jitters[ji]), ji
                break
    return evaded, min_jitter, min_idx


def _stats(evaded, min_jitter):
    got = min_jitter[evaded]
    return {"evasion_rate": float(evaded.mean()),
            "median_min_jitter": float(np.median(got)) if len(got) else None,
            "mean_min_jitter": float(got.mean()) if len(got) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/sweep.npz")
    p.add_argument("--classical", default="results/classical_native")
    p.add_argument("--quantum", default="results/quantum")
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="results/evasion/transfer.json")
    args = p.parse_args()

    sweep = np.load(args.sweep)
    A = Detector.from_result(args.classical, args.data)   # classical MLP
    B = Detector.from_result(args.quantum, args.data)     # VQC
    thr = args.threshold

    evA, mjA, idxA = min_evading_jitter(A, sweep, thr)
    evB, mjB, idxB = min_evading_jitter(B, sweep, thr)

    # paired comparison on the seeds both models evaded (identical flows)
    both = evA & evB
    paired_diff = (mjB[both] - mjA[both])  # >0 => VQC needed more jitter than the MLP
    paired = {
        "n_paired": int(both.sum()),
        "mean_jitter_diff_vqc_minus_mlp": float(paired_diff.mean()) if both.any() else None,
        "vqc_needed_more": int((paired_diff > 0).sum()),
        "vqc_needed_less": int((paired_diff < 0).sum()),
        "equal": int((paired_diff == 0).sum()),
    }

    # transferability: score one model's evading flow with the OTHER model
    def transfer(src_evaded, src_idx, dst):
        hits = tot = 0
        for si in np.where(src_evaded)[0]:
            flow = sweep["features"][si, src_idx[si]]
            tot += 1
            hits += int(dst.prob_benign(flow)[0] >= thr)   # flow is functional by construction
        return {"transfer_rate": hits / tot if tot else None, "n": tot}

    out = {
        "classical": _stats(evA, mjA),
        "quantum": _stats(evB, mjB),
        "paired": paired,
        "transfer_mlp_evasion_to_vqc": transfer(evA, idxA, B),
        "transfer_vqc_evasion_to_mlp": transfer(evB, idxB, A),
        "threshold": thr,
        "jitters": sweep["jitters"].tolist(),
        "n_seeds": int(sweep["features"].shape[0]),
    }
    # print first, so a write failure (e.g. a root-owned results/ dir) never loses the numbers
    print(f"classical MLP: evasion {out['classical']['evasion_rate']:.2f}  "
          f"median jitter {out['classical']['median_min_jitter']}")
    print(f"VQC:           evasion {out['quantum']['evasion_rate']:.2f}  "
          f"median jitter {out['quantum']['median_min_jitter']}")
    print(f"\npaired (n={paired['n_paired']}): mean jitter diff (VQC-MLP) = "
          f"{paired['mean_jitter_diff_vqc_minus_mlp']}  "
          f"[VQC needed more: {paired['vqc_needed_more']}, less: {paired['vqc_needed_less']}, "
          f"equal: {paired['equal']}]")
    print(f"transfer MLP-evasion -> VQC: {out['transfer_mlp_evasion_to_vqc']['transfer_rate']}")
    print(f"transfer VQC-evasion -> MLP: {out['transfer_vqc_evasion_to_mlp']['transfer_rate']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
