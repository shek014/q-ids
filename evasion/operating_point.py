"""Operating-point sensitivity: does evasion survive if the defender moves the decision threshold?

Every detector in the paper flags a flow when P(benign) < 0.5. Here each detector instead gets its own
threshold tau, set on the VALIDATION split so that a target fraction of benign validation flows is
flagged (a target benign false-positive rate). With tau fixed, the saved sweeps are rescored with the
paper's conventions (evasion/paper_metrics.py): naive detection at the grid floor, evasion among detected
attacks, minimal evading value. The achieved FPR and attack recall are reported on the clean test split,
with ROC-AUC as a threshold-free reference. Offline, no captures:

    .venv/bin/python -m evasion.operating_point            # -> results/evasion/operating_point.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evasion import paper_metrics as pm
from evasion.detector import Detector
from features.dataset import load_dataset, train_val_test_split

TARGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
# C2 escalation sweeps rescored at each tuned threshold: payload-size jitter at timing jitter 0.5 (all 30
# instances) and at timing jitter 0.3 (the 9 held-out instances).
C2_ESCALATION = {"size_at_jitter_0.5": "results/evasion/sweep_sizeaxis_full.npz",
                 "size_at_jitter_0.3": "results/evasion/sweep_combined_heldout.npz"}
FAMILIES = {
    "c2": ("data/dataset.npz", "results/evasion/sweep.npz", None,
           [("MLP (native)", "classical_native"), ("MLP (matched)", "classical_matched"), ("VQC", "quantum"),
            ("Random Forest", "rf"), ("XGBoost", "xgb"), ("Rule", "threshold")]),
    "exfil": ("data/exfil_dataset.npz", "results/evasion/exfil_sweep.npz", "completion_time",
              [("MLP (native)", "exfil_classical_native"), ("MLP (matched)", "exfil_classical_matched"),
               ("VQC", "exfil_quantum"), ("Random Forest", "exfil_rf"), ("XGBoost", "exfil_xgb"),
               ("Rule", "exfil_threshold")]),
}


def tau_for(p_benign_val, target):
    """Threshold flagging `target` of the benign validation flows (flag when P(benign) < tau)."""
    return float(np.quantile(p_benign_val, target))


CURVE_TARGETS = [round(x, 3) for x in np.linspace(0.0, 0.6, 31)]


def success(det, sweeps, tau):
    """Per instance (all of them, detected or not): does ANY functional flow across the given sweeps,
    aligned by instance, score P(benign) >= tau? That is the attacker getting through with the
    perturbations those sweeps offer."""
    ok = None
    for s in sweeps:
        P = det.prob_benign_grid(s["features"], s["got"])
        hit = (s["got"] & s["functional"] & (P >= tau)).any(axis=1)
        ok = hit if ok is None else ok | hit
    return float(ok.mean())


def curves():
    """Attack success vs achieved benign test FPR, sweeping tau over validation-benign quantiles."""
    out = {}
    jit, size = np.load("results/evasion/sweep.npz"), np.load(C2_ESCALATION["size_at_jitter_0.5"])
    attacks = {"c2": {"jitter": [jit], "jitter_or_padding": [jit, size]},
               "exfil": {"throttle": [np.load("results/evasion/exfil_sweep.npz")]}}
    for fam, (data, _, _, dets) in FAMILIES.items():
        X, y, _, cn = load_dataset(data)
        _, (Xva, yva), (Xte, yte) = train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=0)
        b = cn.index("benign")
        for name, d in dets:
            det = Detector.from_result(f"results/{d}", data)
            pva, pte = det.prob_benign(Xva[yva == b]), det.prob_benign(Xte[yte == b])
            pts = []
            for t in CURVE_TARGETS:
                tau = tau_for(pva, t) if t > 0 else float(pva.min())
                pts.append({"target": t, "tau": tau, "test_fpr": float((pte < tau).mean()),
                            **{a: success(det, sw, tau) for a, sw in attacks[fam].items()}})
            out.setdefault(fam, {})[d] = {"label": name, "points": pts}
    return out


def run():
    out = {}
    for fam, (data, sweep_path, time_key, dets) in FAMILIES.items():
        X, y, _, cn = load_dataset(data)
        _, (Xva, yva), (Xte, yte) = train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=0)
        b = cn.index("benign")
        sweep = np.load(sweep_path)
        ct = sweep[time_key] if time_key else None
        esc = {k: np.load(v) for k, v in C2_ESCALATION.items()} if fam == "c2" else {}
        print(f"\n===== {fam}: evasion at matched benign FPR (tau set on validation)")
        print(f"{'detector':15s}{'AUC':>6s}  " + "  ".join(f"{h:>26s}" for h in
                                                     ["tau 0.5"] + [f"target FPR {t:.0%}" for t in TARGETS]))
        fam_out = out[fam] = {}
        for name, d in dets:
            det = Detector.from_result(f"results/{d}", data)
            pva, pte = det.prob_benign(Xva), det.prob_benign(Xte)
            mal = yte != b
            auc = float(roc_auc_score(mal, 1 - pte))
            rows = {}
            cells = []
            for t in [None] + TARGETS:
                tau = 0.5 if t is None else tau_for(pva[yva == b], t)
                flag = pte < tau
                r = pm.score(det, sweep, threshold=tau)
                s = pm.summary(r, ct)
                key = "default" if t is None else f"{t:.2f}"
                rows[key] = {"tau": tau, "test_fpr": float(flag[~mal].mean()),
                             "test_recall": float(flag[mal].mean()), **s}
                for ename, es in esc.items():
                    rows[key][ename] = pm.summary(pm.score(det, es, threshold=tau))
                k, n = s["evaded"]
                cells.append(f"fpr{rows[key]['test_fpr']:.2f} det{n:>2d} ev{k:>2d}"
                             + (f" @{s['median_min']:g}" if s["median_min"] is not None else "     "))
            fam_out[d] = {"auc": auc, "operating_points": rows}
            print(f"{name:15s}{auc:6.3f}  " + "  ".join(f"{c:>26s}" for c in cells))
            for ename in esc:
                print(f"{'  ' + ename:21s}  " + "  ".join(
                    f"{'det%2d ev%2d' % (rows[k][ename]['evaded'][1], rows[k][ename]['evaded'][0]) + (' @%g' % rows[k][ename]['median_min'] if rows[k][ename]['median_min'] is not None else ''):>26s}"
                    for k in rows))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/evasion/operating_point.json")
    args = ap.parse_args()
    out = run()
    Path(args.out).write_text(json.dumps({"targets": TARGETS, "results": out, "curves": curves()},
                                         indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
