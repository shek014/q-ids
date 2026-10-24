"""Offline scoring of a capture-once sweep (evasion/sweep.py) against BOTH detectors. Because every
flow is scored by both models, this gives (a) a PAIRED minimal-jitter comparison on identical traffic
and (b) transferability: does one model's evading beacon also fool the other? The sweep now spans
several base configs (check-in intervals) x seeds, so results are also broken down per config, and the
minimal evading jitter is reported as a half-open bracket (last non-evading, first evading] to make the
grid resolution explicit. No captures here, so it runs in the .venv without root:

    .venv/bin/python -m evasion.transfer --sweep results/evasion/sweep.npz
"""
import argparse
import json
import statistics  # noqa: F401 - np.median is used; kept for parity with earlier versions
from pathlib import Path

import numpy as np

from evasion.detector import Detector


def min_evading_jitter(detector, sweep, threshold):
    """Per instance: smallest jitter (ascending) at which the detector calls the flow benign AND it is
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


def _brackets(grid, got, min_idx, evaded):
    """Half-open bracket (low, high] for each evading instance: high = the first evading grid value,
    low = the last CAPTURED non-evading value below it (0.0 if none / evaded at the grid floor). Makes
    the reported minimum a measured interval rather than a point pinned to grid resolution."""
    n = len(min_idx)
    low = np.full(n, np.nan)
    high = np.full(n, np.nan)
    for i in range(n):
        if not evaded[i]:
            continue
        hi = int(min_idx[i])
        high[i] = float(grid[hi])
        lo = 0.0
        for k in range(hi - 1, -1, -1):
            if got[i, k]:
                lo = float(grid[k])
                break
        low[i] = lo
    return low, high


def _median_bracket(low, high, evaded):
    if not evaded.any():
        return None
    return [float(np.median(low[evaded])), float(np.median(high[evaded]))]


def _per_config(sweep, evaded, low, high, key):
    """Headline metrics grouped by base config, labelled by the varied field `key` (e.g. interval)."""
    if "inst_config_idx" not in sweep:
        return None
    cfg_idx = sweep["inst_config_idx"]
    table = [json.loads(s) for s in sweep["config_table"]]
    out = []
    for c, cfg in enumerate(table):
        mask = cfg_idx == c
        ev = evaded & mask
        out.append({
            "label": f"{key}={cfg.get(key)}",
            "config": cfg,
            "n": int(mask.sum()),
            "evasion_rate": float(evaded[mask].mean()) if mask.any() else None,
            "median_min_jitter": float(np.median(high[ev])) if ev.any() else None,
            "median_bracket": [float(np.median(low[ev])), float(np.median(high[ev]))] if ev.any() else None,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/sweep.npz")
    p.add_argument("--classical", default="results/classical_native")
    p.add_argument("--quantum", default="results/quantum")
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--config-key", default="interval", help="varied base-config field, for per-config labels")
    p.add_argument("--out", default="results/evasion/transfer.json")
    args = p.parse_args()

    sweep = np.load(args.sweep)
    A = Detector.from_result(args.classical, args.data)   # classical MLP
    B = Detector.from_result(args.quantum, args.data)     # VQC
    thr = args.threshold
    grid, got = sweep["jitters"], sweep["got"]

    evA, mjA, idxA = min_evading_jitter(A, sweep, thr)
    evB, mjB, idxB = min_evading_jitter(B, sweep, thr)
    loA, hiA = _brackets(grid, got, idxA, evA)
    loB, hiB = _brackets(grid, got, idxB, evB)

    # paired comparison on the instances both models evaded (identical flows). Classify in GRID STEPS
    # so a "tie" is explicitly indistinguishable-at-resolution, not a claim of exact equality.
    both = evA & evB
    diff_idx = idxB[both] - idxA[both]           # >0 => VQC needed a higher grid step than the MLP
    paired = {
        "n_paired": int(both.sum()),
        "mean_jitter_diff_vqc_minus_mlp": float((mjB[both] - mjA[both]).mean()) if both.any() else None,
        "vqc_harder": int((diff_idx > 0).sum()),
        "mlp_harder": int((diff_idx < 0).sum()),
        "tie_at_grid_resolution": int((diff_idx == 0).sum()),
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
        "classical": {**_stats(evA, mjA), "median_bracket": _median_bracket(loA, hiA, evA),
                      "per_config": _per_config(sweep, evA, loA, hiA, args.config_key)},
        "quantum": {**_stats(evB, mjB), "median_bracket": _median_bracket(loB, hiB, evB),
                    "per_config": _per_config(sweep, evB, loB, hiB, args.config_key)},
        "paired": paired,
        "transfer_mlp_evasion_to_vqc": transfer(evA, idxA, B),
        "transfer_vqc_evasion_to_mlp": transfer(evB, idxB, A),
        "threshold": thr,
        "jitters": grid.tolist(),
        "n_instances": int(sweep["features"].shape[0]),
    }
    # print first, so a write failure (e.g. a root-owned results/ dir) never loses the numbers
    print(f"classical MLP: evasion {out['classical']['evasion_rate']:.2f}  "
          f"median jitter {out['classical']['median_min_jitter']}  bracket {out['classical']['median_bracket']}")
    print(f"VQC:           evasion {out['quantum']['evasion_rate']:.2f}  "
          f"median jitter {out['quantum']['median_min_jitter']}  bracket {out['quantum']['median_bracket']}")
    if out["classical"]["per_config"]:
        print("per-config (classical MLP):")
        for row in out["classical"]["per_config"]:
            print(f"  {row['label']:16s} n={row['n']} evasion {row['evasion_rate']:.2f}  "
                  f"median jitter {row['median_min_jitter']}  bracket {row['median_bracket']}")
    print(f"\npaired (n={paired['n_paired']}): mean jitter diff (VQC-MLP) = "
          f"{paired['mean_jitter_diff_vqc_minus_mlp']}  "
          f"[VQC harder: {paired['vqc_harder']}, MLP harder: {paired['mlp_harder']}, "
          f"tie@grid: {paired['tie_at_grid_resolution']}]")
    print(f"transfer MLP-evasion -> VQC: {out['transfer_mlp_evasion_to_vqc']['transfer_rate']}")
    print(f"transfer VQC-evasion -> MLP: {out['transfer_vqc_evasion_to_mlp']['transfer_rate']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
