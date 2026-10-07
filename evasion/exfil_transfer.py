"""Offline scoring of an EXFIL capture-once sweep (evasion/exfil_sweep.py) against all three detectors
(native MLP, matched MLP, VQC). Because every flow is scored by every model, this gives (a) each
detector's throttle-evasion rate, (b) a PAIRED minimal-throttle comparison of the matched MLP vs the
VQC on identical traffic (the fair matched-input comparison), and (c) transferability between the
deployed native MLP and the VQC. No captures here, so it runs in the .venv without root. Mirrors
evasion/transfer.py (the C2 version):

    .venv/bin/python -m evasion.exfil_transfer --sweep results/evasion/exfil_sweep.npz
"""
import argparse
import json
import statistics  # noqa: F401 - kept parallel to transfer.py; np.median is used below
from pathlib import Path

import numpy as np

from evasion.detector import Detector


def min_evading_throttle(detector, sweep, threshold):
    """Per seed: smallest throttle (ascending) at which the detector calls the flow benign AND the
    exfil is still functional (completed within the deadline). Returns (evaded, min_throttle, min_idx)."""
    features, functional, got, throttles = (sweep["features"], sweep["functional"],
                                            sweep["got"], sweep["throttles"])
    n_s, n_t, _ = features.shape
    evaded = np.zeros(n_s, bool)
    min_throttle = np.full(n_s, np.inf)
    min_idx = np.full(n_s, -1, int)
    for si in range(n_s):
        for ti in range(n_t):
            if not got[si, ti]:
                continue
            if functional[si, ti] and detector.prob_benign(features[si, ti])[0] >= threshold:
                evaded[si], min_throttle[si], min_idx[si] = True, float(throttles[ti]), ti
                break
    return evaded, min_throttle, min_idx


def _stats(evaded, min_throttle):
    got = min_throttle[evaded]
    return {"evasion_rate": float(evaded.mean()),
            "median_min_throttle": float(np.median(got)) if len(got) else None,
            "mean_min_throttle": float(got.mean()) if len(got) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/exfil_sweep.npz")
    p.add_argument("--native", default="results/exfil_classical_native")
    p.add_argument("--matched", default="results/exfil_classical_matched")
    p.add_argument("--quantum", default="results/exfil_quantum")
    p.add_argument("--data", default="data/exfil_dataset.npz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="results/evasion/exfil_transfer.json")
    args = p.parse_args()

    sweep = np.load(args.sweep)
    N = Detector.from_result(args.native, args.data)    # native MLP (deployed)
    M = Detector.from_result(args.matched, args.data)   # matched MLP (PCA-8, fair vs VQC)
    Q = Detector.from_result(args.quantum, args.data)   # VQC
    thr = args.threshold

    evN, mjN, idxN = min_evading_throttle(N, sweep, thr)
    evM, mjM, idxM = min_evading_throttle(M, sweep, thr)
    evQ, mjQ, idxQ = min_evading_throttle(Q, sweep, thr)

    # paired comparison on matched inputs (matched MLP vs VQC), on seeds both evaded (identical flows)
    both = evM & evQ
    paired_diff = (mjQ[both] - mjM[both])  # >0 => VQC needed more throttle than the matched MLP
    paired = {
        "comparison": "matched_mlp_vs_vqc",
        "n_paired": int(both.sum()),
        "mean_throttle_diff_vqc_minus_mlp": float(paired_diff.mean()) if both.any() else None,
        "vqc_needed_more": int((paired_diff > 0).sum()),
        "vqc_needed_less": int((paired_diff < 0).sum()),
        "equal": int((paired_diff == 0).sum()),
    }

    # transferability: score one model's evading flow with the OTHER model (deployed MLP <-> VQC)
    def transfer(src_evaded, src_idx, dst):
        hits = tot = 0
        for si in np.where(src_evaded)[0]:
            flow = sweep["features"][si, src_idx[si]]
            tot += 1
            hits += int(dst.prob_benign(flow)[0] >= thr)   # flow is functional by construction
        return {"transfer_rate": hits / tot if tot else None, "n": tot}

    out = {
        "native_mlp": _stats(evN, mjN),
        "matched_mlp": _stats(evM, mjM),
        "quantum": _stats(evQ, mjQ),
        "paired": paired,
        "transfer_mlp_evasion_to_vqc": transfer(evN, idxN, Q),
        "transfer_vqc_evasion_to_mlp": transfer(evQ, idxQ, N),
        "threshold": thr,
        "throttles": sweep["throttles"].tolist(),
        "n_seeds": int(sweep["features"].shape[0]),
    }
    # print first, so a write failure (e.g. a root-owned results/ dir) never loses the numbers
    print(f"native MLP:  evasion {out['native_mlp']['evasion_rate']:.2f}  "
          f"median throttle {out['native_mlp']['median_min_throttle']}")
    print(f"matched MLP: evasion {out['matched_mlp']['evasion_rate']:.2f}  "
          f"median throttle {out['matched_mlp']['median_min_throttle']}")
    print(f"VQC:         evasion {out['quantum']['evasion_rate']:.2f}  "
          f"median throttle {out['quantum']['median_min_throttle']}")
    print(f"\npaired matched-MLP vs VQC (n={paired['n_paired']}): mean throttle diff (VQC-MLP) = "
          f"{paired['mean_throttle_diff_vqc_minus_mlp']}  "
          f"[VQC more: {paired['vqc_needed_more']}, less: {paired['vqc_needed_less']}, "
          f"equal: {paired['equal']}]")
    print(f"transfer MLP-evasion -> VQC: {out['transfer_mlp_evasion_to_vqc']['transfer_rate']}")
    print(f"transfer VQC-evasion -> MLP: {out['transfer_vqc_evasion_to_mlp']['transfer_rate']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
