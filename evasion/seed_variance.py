"""Training-seed variance: retrain every detector with several model seeds and rescore the saved sweeps,
offline (no captures, no root). The data split stays fixed (split seed 0, the paper's split); only the
model seed changes: MLP/VQC initialisation and batch order, Random Forest and XGBoost random_state. The
single-feature rule is deterministic, so it is not retrained. Seed 0 reproduces the paper's models.

Per seed, for both families, it trains the clean detectors, the single-axis hardened detectors and (C2)
the jointly hardened detectors exactly as evasion/harden.py, evasion/exfil_harden.py and
evasion/joint_harden.py do, scores them with the conventions of evasion/paper_metrics.py, and writes
results/seeds/seed_<k>.json. Finished seeds are skipped, so runs can be split across processes:

    .venv/bin/python -m evasion.seed_variance --seeds 0-4 &
    .venv/bin/python -m evasion.seed_variance --seeds 5-9 &
    .venv/bin/python -m evasion.seed_variance --summarize      # median [min, max] across seeds
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from evasion import paper_metrics as pm
from evasion.detector import Detector
from evasion.harden import adversarial_flows, parse_seeds, split_instances, train_mlp, train_vqc
from features.dataset import load_dataset, train_val_test_split

OUT = Path("results/seeds")
EV = Path("results/evasion")
LEARNED = ["mlp_native", "mlp_matched", "vqc", "rf", "xgb"]


def splits_for(data):
    X, y, _, class_names = load_dataset(data)
    return train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=0), class_names


def augment(splits, extra, attack_idx):
    (Xtr, ytr), va, te = splits
    if extra is None:
        return splits
    return (np.vstack([Xtr, extra]), np.concatenate([ytr, np.full(len(extra), attack_idx)])), va, te


def train_all(splits, class_names, seed, cfgs, tmp):
    """Train the five learned detectors into tmp/<name> with model seed `seed`; returns {name: Detector}."""
    from models.baselines.train import fit_baseline
    mlp_cfg, vqc_cfg, b_cfg = ({**c, "seed": seed} for c in cfgs)
    train_mlp(splits, None, mlp_cfg, class_names, tmp / "mlp_native")
    train_mlp(splits, vqc_cfg["n_qubits"], mlp_cfg, class_names, tmp / "mlp_matched")
    train_vqc(splits, vqc_cfg, class_names, tmp / "vqc")
    fit_baseline("random_forest", splits, b_cfg, class_names, tmp / "rf")
    fit_baseline("xgboost", splits, b_cfg, class_names, tmp / "xgb")
    return {n: Detector.from_result(tmp / n) for n in LEARNED}


def clean(det, splits, class_names):
    """Clean metrics on the fixed test split (same numbers as paper_metrics.clean_metrics)."""
    _, _, (Xte, yte) = splits
    flag = det.prob_benign(Xte) < pm.THRESHOLD
    mal = yte != class_names.index("benign")
    return {"accuracy": float((flag == mal).mean()),
            "recall": float((flag & mal).sum() / mal.sum()),
            "benign_fpr": float((flag & ~mal).sum() / (~mal).sum())}


def c2_seed(seed, cfgs, tmp):
    splits, cn = splits_for("data/dataset.npz")
    a = cn.index("c2")
    jit = np.load(EV / "sweep.npz")
    size = np.load(EV / "sweep_sizeaxis_full.npz")
    comb = np.load(EV / "sweep_combined_heldout.npz")
    jtr, held = split_instances(jit, 0.67)
    s_tr, _ = split_instances(size, 0.67)
    adv_j = adversarial_flows(jit, jtr)
    adv_js = np.vstack([adv_j, adversarial_flows(size, s_tr)])
    out = {}
    for variant, extra in [("clean", None), ("hardened", adv_j), ("joint", adv_js)]:
        sp = augment(splits, extra, a)
        dets = train_all(sp, cn, seed, cfgs, tmp / variant)
        v = out[variant] = {}
        res = {}
        for n, d in dets.items():
            v[n] = {"clean": clean(d, sp, cn)}
            if variant == "clean":
                res[n] = pm.score(d, jit)
                v[n]["jitter_all"] = pm.summary(res[n])
            else:
                v[n]["jitter_held"] = pm.summary(pm.score(d, jit, held))
                v[n]["size_held"] = pm.summary(pm.score(d, size, held))
                v[n]["combined_held"] = pm.summary(pm.score(d, comb))
        if variant == "clean":
            v["paired_matched_vqc"] = pm.paired(res["mlp_matched"], res["vqc"])
            v["paired_native_vqc"] = pm.paired(res["mlp_native"], res["vqc"])
            v["transfer"] = {f"{s}->{t}": pm.transfer(res[s], res[t]) for s, t in
                             [("mlp_native", "vqc"), ("vqc", "mlp_native"), ("mlp_native", "rf"),
                              ("mlp_native", "xgb")]}
    return out


def exfil_seed(seed, cfgs, tmp):
    splits, cn = splits_for("data/exfil_dataset.npz")
    a = cn.index("exfil")
    sw = np.load(EV / "exfil_sweep.npz")
    ct = sw["completion_time"]
    tr, held = split_instances(sw, 0.67)
    esc = {t: np.load(EV / f"exfil_esc_{t}.npz") for t in ("volume", "jitter", "chunk")}
    out = {}
    for variant, extra in [("clean", None), ("hardened", adversarial_flows(sw, tr))]:
        sp = augment(splits, extra, a)
        dets = train_all(sp, cn, seed, cfgs, tmp / ("exfil_" + variant))
        v = out[variant] = {}
        res = {}
        for n, d in dets.items():
            v[n] = {"clean": clean(d, sp, cn)}
            if variant == "clean":
                res[n] = pm.score(d, sw)
                v[n]["throttle_all"] = pm.summary(res[n], ct)
            else:
                v[n]["throttle_held"] = pm.summary(pm.score(d, sw, held), ct[held])
                for t, s in esc.items():
                    v[n][f"esc_{t}"] = pm.summary(pm.score(d, s), s["completion_time"])
        if variant == "clean":
            v["paired_matched_vqc"] = pm.paired(res["mlp_matched"], res["vqc"])
            v["transfer"] = {f"{s}->{t}": pm.transfer(res[s], res[t]) for s, t in
                             [("mlp_native", "vqc"), ("vqc", "mlp_native")]}
    return out


def run(seeds):
    cfgs = tuple(yaml.safe_load(open(p)) for p in
                 ("configs/mlp.yaml", "configs/vqc.yaml", "configs/baselines.yaml"))
    OUT.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        path = OUT / f"seed_{s}.json"
        if path.exists():
            print(f"seed {s}: done, skipping")
            continue
        t0 = time.time()
        tmp = Path(tempfile.mkdtemp(prefix=f"seedvar_{s}_"))
        try:
            rec = {"seed": s, "c2": c2_seed(s, cfgs, tmp), "exfil": exfil_seed(s, cfgs, tmp)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        path.write_text(json.dumps(rec, indent=1, default=float))
        print(f"seed {s}: wrote {path} in {time.time() - t0:.0f} s", flush=True)


def _get(rec, keys):
    for k in keys:
        rec = rec[k]
    return rec


def summarize():
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("seed_*.json"))]
    seeds = [r["seed"] for r in recs]
    print(f"{len(recs)} seeds: {seeds}\n")

    def stat(vals):
        vals = [x for x in vals if x is not None]
        if not vals:
            return "-"
        return f"{np.median(vals):.3g} [{min(vals):.3g}, {max(vals):.3g}]"

    def frac(rows):  # evaded/detected per seed -> median rate [min, max]
        return stat([r["evaded"][0] / r["evaded"][1] if r["evaded"][1] else None for r in rows])

    summ = {}
    for fam, variants in [("c2", {"clean": ["jitter_all"],
                                  "hardened": ["jitter_held", "size_held", "combined_held"],
                                  "joint": ["jitter_held", "size_held", "combined_held"]}),
                          ("exfil", {"clean": ["throttle_all"],
                                     "hardened": ["throttle_held", "esc_volume", "esc_jitter", "esc_chunk"]})]:
        for var, lines in variants.items():
            print(f"== {fam} / {var}: accuracy | benign FPR | evasion rate | median min value | time")
            for n in LEARNED:
                acc = stat([_get(r, [fam, var, n, "clean", "accuracy"]) for r in recs])
                fpr = stat([_get(r, [fam, var, n, "clean", "benign_fpr"]) for r in recs])
                print(f"  {n:12s} acc {acc}   FPR {fpr}")
                for ln in lines:
                    rows = [_get(r, [fam, var, n, ln]) for r in recs]
                    tm = stat([x.get("median_time") for x in rows]) if fam == "exfil" else ""
                    line = f"{frac(rows)}  min {stat([x['median_min'] for x in rows])}  {tm}"
                    summ.setdefault(fam, {}).setdefault(var, {}).setdefault(n, {})[ln] = line
                    print(f"      {ln:14s} evaded {line}")
        for key in ("paired_matched_vqc", "paired_native_vqc"):
            if key in recs[0][fam]["clean"]:
                ps = [r[fam]["clean"][key] for r in recs]
                print(f"  {fam} {key}: per seed (n, VQC higher, VQC lower, tie): "
                      + "; ".join(f"s{r['seed']} {p['n']},{p['b_higher']},{p['b_lower']},{p['tie']}"
                                  for r, p in zip(recs, ps)))
        for t in recs[0][fam]["clean"]["transfer"]:
            vals = [r[fam]["clean"]["transfer"][t]["transferred"] for r in recs]
            print(f"  {fam} transfer {t}: " + " ".join(f"{k}/{n}" for k, n in vals))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-9", help="model seeds, e.g. 0-9 or 0,3,5")
    ap.add_argument("--summarize", action="store_true", help="print median [min, max] across finished seeds")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run(parse_seeds(args.seeds))


if __name__ == "__main__":
    main()
