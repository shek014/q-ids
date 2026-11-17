"""Score the 10-minute C2 runs from evasion/long_run.py, offline. Three questions:

1. Alarm aggregation. Each detector scores every 60 s window of every flow, exactly as it scores a
   training flow. A host-level alarm fires when at least k of the flow's windows are flagged
   (k = 1..10), and alternatively when the mean P(benign) over windows falls below a threshold set for
   a target benign false-positive rate. Does aggregating over ten windows catch a jittered or padded
   beacon that single windows miss, at a false-positive rate a defender would accept?
2. A RITA-style beacon scorer (evasion/rita_score.py) over the whole run, the long-horizon detector
   practitioners use, compared at matched benign false-positive rates.
3. The observation-window argument: cv_iat of each UDP flow over the whole run against its value in
   single 60 s windows.

Units: one flow per (run, host). Beacon flows are the controlled beacon h2 at each setting (9 per
setting); the scenario's own naive beacon h3 runs at training parameters in every run; benign flows are
h4-h8 of every run. The background of an instance is the same at every setting, so benign flows repeat
their parameters across the settings of one instance.

    .venv/bin/python -m evasion.long_eval                    # -> results/long/long_eval.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evasion import rita_score
from evasion.detector import Detector
from features.dataset import FEATURE_NAMES

DETECTORS = [("MLP (native)", "classical_native"), ("MLP (matched)", "classical_matched"),
             ("VQC", "quantum"), ("Random Forest", "rf"), ("XGBoost", "xgb"), ("Rule", "threshold")]
TARGETS = [0.01, 0.05, 0.10, 0.20]


def load_runs(run_dir):
    flows = []
    for path in sorted(Path(run_dir).glob("inst*.npz")):
        if ".tmp" in path.name:
            continue
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        th = meta["theta"]
        setting = (float(th["jitter"]), float(th["size_jitter"]))
        for fi, host in enumerate(d["flow_host"]):
            c = d["conn_flow"] == fi
            pk = d["pkt_flow"] == fi
            flows.append({"run": path.stem, "seed": int(meta["seed"]), "setting": setting,
                          "host": str(host), "role": str(d["flow_role"][fi]), "proto": str(d["flow_proto"][fi]),
                          "functional": bool(d["functional"]),
                          "win": d["win_features"][fi], "npk": d["win_npk"][fi],
                          "conn_ts": d["conn_ts"][c], "conn_bytes": d["conn_bytes"][c],
                          "pkt_ts": d["pkt_ts"][pk]})
    return flows


def groups(flows):
    """Index sets: 'benign', 'naive' (h3), and one per beacon setting."""
    g = defaultdict(list)
    for i, f in enumerate(flows):
        if f["role"] == "benign":
            g["benign"].append(i)
        elif f["host"] == "h3":
            g["naive"].append(i)
        elif f["role"] == "beacon":
            g[f"beacon {f['setting'][0]:g}/{f['setting'][1]:g}"].append(i)
    return {k: np.array(v) for k, v in g.items()}


def at_fpr(score, benign, target):
    """Higher score = more suspicious. Lowest threshold whose benign FPR (score >= thr) is <= target.
    Returns (threshold, achieved FPR)."""
    b = np.sort(score[benign])[::-1]
    cands = np.unique(np.concatenate([score[np.isfinite(score)], [np.inf]]))[::-1]
    best = (np.inf, 0.0)
    for thr in cands:
        fpr = float((b >= thr).mean())
        if fpr > target:
            break
        best = (float(thr), fpr)
    return best


def rate(mask):
    return [int(mask.sum()), int(mask.size)]


def windowed(flows, G, dets):
    out = {}
    for label, d in dets:
        det = Detector.from_result(f"results/{d}", "data/dataset.npz")
        nflag, nval, meanP = [], [], []
        for f in flows:
            ok = f["npk"] > 0
            P = det.prob_benign(f["win"][ok]) if ok.any() else np.array([])
            nflag.append(int((P < 0.5).sum()))
            nval.append(int(ok.sum()))
            meanP.append(float(P.mean()) if len(P) else 1.0)
        nflag, nval, meanP = np.array(nflag), np.array(nval), np.array(meanP)
        rec = {"label": label, "window_flag_rate": {}, "k_of_n": {}, "mean_p": {}}
        for g, idx in G.items():
            rec["window_flag_rate"][g] = [int(nflag[idx].sum()), int(nval[idx].sum())]
        for k in range(1, 11):
            rec["k_of_n"][k] = {g: rate(nflag[idx] >= k) for g, idx in G.items()}
        susp = 1.0 - meanP
        for t in TARGETS:
            thr, fpr = at_fpr(susp, G["benign"], t)
            rec["mean_p"][t] = {"threshold": thr, "benign_fpr": fpr,
                                **{g: rate(susp[idx] >= thr) for g, idx in G.items() if g != "benign"}}
        out[d] = rec
    return out


def rita(flows, G):
    """RITA-style score per flow over the whole run. The false-positive rate is taken over the periodic
    UDP benign hosts (h4-h6) only: the bursty TCP hosts (h7, h8) are restarted on an exact 60 s schedule
    by the testbed, which a beacon scorer correctly reads as periodic, so their scores describe the
    scheduler, not the traffic. Their medians are still reported under benign_by_host."""
    scores = np.array([rita_score.beacon_score(f["conn_ts"], f["conn_bytes"]) for f in flows])
    s = scores[:, 0]
    udp_benign = np.array([i for i in G["benign"] if flows[i]["proto"] == "udp"])
    rec = {"median_score": {g: float(np.median(s[idx])) for g, idx in G.items()},
           "median_ts_score": {g: float(np.median(scores[idx, 1])) for g, idx in G.items()},
           "median_ds_score": {g: float(np.median(scores[idx, 2])) for g, idx in G.items()},
           "benign_by_host": {}, "at_fpr": {}, "fpr_pool": "udp benign (h4-h6)", "n_fpr_pool": len(udp_benign)}
    for h in sorted({flows[i]["host"] for i in G["benign"]}):
        idx = np.array([i for i in G["benign"] if flows[i]["host"] == h])
        rec["benign_by_host"][h] = float(np.median(s[idx]))
    for t in TARGETS:
        thr, fpr = at_fpr(s, udp_benign, t)
        rec["at_fpr"][t] = {"threshold": thr, "benign_fpr": fpr,
                            **{g: rate(s[idx] >= thr) for g, idx in G.items() if g != "benign"}}
    return rec, s


def cv_iat(flows, G):
    """cv_iat over the whole run vs the median over its 60 s windows, for UDP flows."""
    ci = FEATURE_NAMES.index("cv_iat")
    rec = {}
    by = defaultdict(lambda: ([], []))
    for g, idx in G.items():
        for i in idx:
            f = flows[i]
            if f["proto"] != "udp" or len(f["pkt_ts"]) < 3:
                continue
            iat = np.diff(np.sort(f["pkt_ts"]))
            key = g if g != "benign" else f"benign {f['host']}"
            by[key][0].append(float(iat.std() / iat.mean()))
            by[key][1].append(float(np.nanmedian(f["win"][f["npk"] > 1, ci])))
    for k, (full, win) in sorted(by.items()):
        rec[k] = {"n": len(full), "full_run_median": float(np.median(full)),
                  "window_median": float(np.median(win)),
                  "full_run_range": [float(min(full)), float(max(full))]}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/long")
    ap.add_argument("--out", default="results/long/long_eval.json")
    ap.add_argument("--detectors", default=None,
                    help="comma-separated result dirs under results/ (default: the six clean detectors)")
    args = ap.parse_args()
    dets = DETECTORS if not args.detectors else [(d, d) for d in args.detectors.split(",")]
    flows = load_runs(args.runs)
    G = groups(flows)
    beacons = sorted(g for g in G if g.startswith("beacon"))
    order = ["benign", "naive"] + beacons
    runs = {f["run"]: f["functional"] for f in flows}
    print(f"{len(runs)} runs, {sum(runs.values())} with a functional beacon; flows: "
          + ", ".join(f"{g} {len(G[g])}" for g in order if g in G))

    out = {"groups": {g: len(G[g]) for g in G}, "functional_runs": [sum(runs.values()), len(runs)]}
    out["windowed"] = windowed(flows, G, dets)
    print("\n== single 60 s windows flagged (flagged/windows)")
    for d, rec in out["windowed"].items():
        print(f"  {rec['label']:14s} " + "  ".join(f"{g} {a}/{b}" for g in order if g in G
                                                  for a, b in [rec["window_flag_rate"][g]]))
    print("\n== k-of-10 window alarm, flows alarmed (benign | naive | beacon settings)")
    for d, rec in out["windowed"].items():
        print(f"  {rec['label']}")
        for k in (1, 3, 5, 8, 10):
            print(f"    k={k:<2d} " + "  ".join(f"{g} {a}/{b}" for g in order if g in G
                                               for a, b in [rec["k_of_n"][k][g]]))
    print("\n== mean P(benign) over windows, threshold at target benign FPR (detected/flows)")
    for d, rec in out["windowed"].items():
        print(f"  {rec['label']}")
        for t, r in rec["mean_p"].items():
            print(f"    FPR<= {t:<4g} (got {r['benign_fpr']:.3f})  " + "  ".join(
                f"{g} {r[g][0]}/{r[g][1]}" for g in order if g in r))
    out["rita"], _ = rita(flows, G)
    print(f"\n== RITA-style score over the whole run (FPR over {out['rita']['n_fpr_pool']} UDP benign flows): medians " + "  ".join(
        f"{g} {out['rita']['median_score'][g]:.3f}" for g in order if g in G))
    print("   benign medians by host: " + "  ".join(f"{h} {v:.3f}" for h, v in out["rita"]["benign_by_host"].items()))
    for t, r in out["rita"]["at_fpr"].items():
        print(f"    FPR<= {t:<4g} (got {r['benign_fpr']:.3f}, thr {r['threshold']:.3f})  " + "  ".join(
            f"{g} {r[g][0]}/{r[g][1]}" for g in order if g in r))
    out["cv_iat"] = cv_iat(flows, G)
    print("\n== cv_iat: whole run vs median 60 s window (UDP flows)")
    for k, r in out["cv_iat"].items():
        print(f"    {k:22s} n={r['n']:<3d} whole {r['full_run_median']:.3f}  window {r['window_median']:.3f}  "
              f"range [{r['full_run_range'][0]:.3f}, {r['full_run_range'][1]:.3f}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
