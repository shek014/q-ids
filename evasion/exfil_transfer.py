"""Offline scoring of an EXFIL capture-once sweep (evasion/exfil_sweep.py) against all three detectors
(native MLP, matched MLP, VQC). Because every flow is scored by every model, this gives (a) each
detector's throttle-evasion rate, (b) a PAIRED minimal-throttle comparison of the matched MLP vs the
VQC on identical traffic, and (c) transferability between the deployed native MLP and the VQC. The
sweep now spans several base configs (payload volumes) x seeds, so results are also broken down per
config, and the minimal evading throttle is reported as a half-open bracket to make grid resolution
explicit. No captures here, so it runs in the .venv without root. Mirrors evasion/transfer.py:

    .venv/bin/python -m evasion.exfil_transfer --sweep results/evasion/exfil_sweep.npz
"""
import argparse
import json
import statistics  # noqa: F401 - np.median is used; kept for parity with transfer.py
from pathlib import Path

import numpy as np

from evasion.detector import Detector


def min_evading_throttle(detector, sweep, threshold):
    """Per instance: smallest throttle (ascending) at which the detector calls the flow benign AND the
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


def _brackets(grid, got, min_idx, evaded):
    """Half-open bracket (low, high] per evading instance (see evasion/transfer.py for the rationale)."""
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
    """Headline metrics grouped by base config, labelled by the varied field `key` (e.g. volume)."""
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
            "median_min_throttle": float(np.median(high[ev])) if ev.any() else None,
            "median_bracket": [float(np.median(low[ev])), float(np.median(high[ev]))] if ev.any() else None,
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/exfil_sweep.npz")
    p.add_argument("--native", default="results/exfil_classical_native")
    p.add_argument("--matched", default="results/exfil_classical_matched")
    p.add_argument("--quantum", default="results/exfil_quantum")
    p.add_argument("--data", default="data/exfil_dataset.npz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--config-key", default="volume", help="varied base-config field, for per-config labels")
    p.add_argument("--out", default="results/evasion/exfil_transfer.json")
    args = p.parse_args()

    sweep = np.load(args.sweep)
    N = Detector.from_result(args.native, args.data)    # native MLP (deployed)
    M = Detector.from_result(args.matched, args.data)   # matched MLP (PCA-8, fair vs VQC)
    Q = Detector.from_result(args.quantum, args.data)   # VQC
    thr = args.threshold
    grid, got = sweep["throttles"], sweep["got"]

    evN, mjN, idxN = min_evading_throttle(N, sweep, thr)
    evM, mjM, idxM = min_evading_throttle(M, sweep, thr)
    evQ, mjQ, idxQ = min_evading_throttle(Q, sweep, thr)
    loN, hiN = _brackets(grid, got, idxN, evN)
    loM, hiM = _brackets(grid, got, idxM, evM)
    loQ, hiQ = _brackets(grid, got, idxQ, evQ)

    # paired comparison on matched inputs (matched MLP vs VQC), on instances both evaded (identical
    # flows). Classify in GRID STEPS so a "tie" is indistinguishable-at-resolution, not exact equality.
    both = evM & evQ
    diff_idx = idxQ[both] - idxM[both]           # >0 => VQC needed a higher grid step than matched MLP
    paired = {
        "comparison": "matched_mlp_vs_vqc",
        "n_paired": int(both.sum()),
        "mean_throttle_diff_vqc_minus_mlp": float((mjQ[both] - mjM[both]).mean()) if both.any() else None,
        "vqc_harder": int((diff_idx > 0).sum()),
        "mlp_harder": int((diff_idx < 0).sum()),
        "tie_at_grid_resolution": int((diff_idx == 0).sum()),
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
        "native_mlp": {**_stats(evN, mjN), "median_bracket": _median_bracket(loN, hiN, evN),
                       "per_config": _per_config(sweep, evN, loN, hiN, args.config_key)},
        "matched_mlp": {**_stats(evM, mjM), "median_bracket": _median_bracket(loM, hiM, evM),
                        "per_config": _per_config(sweep, evM, loM, hiM, args.config_key)},
        "quantum": {**_stats(evQ, mjQ), "median_bracket": _median_bracket(loQ, hiQ, evQ),
                    "per_config": _per_config(sweep, evQ, loQ, hiQ, args.config_key)},
        "paired": paired,
        "transfer_mlp_evasion_to_vqc": transfer(evN, idxN, Q),
        "transfer_vqc_evasion_to_mlp": transfer(evQ, idxQ, N),
        "threshold": thr,
        "throttles": grid.tolist(),
        "n_instances": int(sweep["features"].shape[0]),
    }
    # print first, so a write failure (e.g. a root-owned results/ dir) never loses the numbers
    print(f"native MLP:  evasion {out['native_mlp']['evasion_rate']:.2f}  "
          f"median throttle {out['native_mlp']['median_min_throttle']}  bracket {out['native_mlp']['median_bracket']}")
    print(f"matched MLP: evasion {out['matched_mlp']['evasion_rate']:.2f}  "
          f"median throttle {out['matched_mlp']['median_min_throttle']}  bracket {out['matched_mlp']['median_bracket']}")
    print(f"VQC:         evasion {out['quantum']['evasion_rate']:.2f}  "
          f"median throttle {out['quantum']['median_min_throttle']}  bracket {out['quantum']['median_bracket']}")
    if out["matched_mlp"]["per_config"]:
        print("per-config (matched MLP):")
        for row in out["matched_mlp"]["per_config"]:
            print(f"  {row['label']:18s} n={row['n']} evasion {row['evasion_rate']:.2f}  "
                  f"median throttle {row['median_min_throttle']}  bracket {row['median_bracket']}")
    print(f"\npaired matched-MLP vs VQC (n={paired['n_paired']}): mean throttle diff (VQC-MLP) = "
          f"{paired['mean_throttle_diff_vqc_minus_mlp']}  "
          f"[VQC harder: {paired['vqc_harder']}, MLP harder: {paired['mlp_harder']}, "
          f"tie@grid: {paired['tie_at_grid_resolution']}]")
    print(f"transfer MLP-evasion -> VQC: {out['transfer_mlp_evasion_to_vqc']['transfer_rate']}")
    print(f"transfer VQC-evasion -> MLP: {out['transfer_vqc_evasion_to_mlp']['transfer_rate']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
