"""Evaluate the joint timing+size defense round against single-axis hardening and the undefended
detectors, on held-out instances, offline (no captures, no root):

    .venv/bin/python -m evasion.joint_eval

For every detector it reports held-out evasion on three axes —
  (a) timing jitter            : results/evasion/sweep.npz            (held-out instances)
  (b) payload size             : results/evasion/sweep_sizeaxis_full.npz (held-out instances)
  (c) combined, untouched      : results/evasion/sweep_combined_heldout.npz
        (timing jitter held at 0.3 — a value neither hardening line trained on — while size varies, so
        the (jitter=0.3, size>0) interior is a region joint hardening never saw directly)
— for the undefended model, the single-axis (jitter) hardened model, and the joint (timing+size)
hardened model, plus each model's clean-test accuracy and benign FPR from its meta.json. Writes
results/evasion/joint_harden_eval.json.
"""
import json
from pathlib import Path

import numpy as np

from evasion.detector import Detector
from evasion.transfer import min_evading_jitter
from evasion.harden import split_instances

THRESH = 0.5
BASES = [("classical_native", "MLP-native"), ("classical_matched", "MLP-matched"),
         ("quantum", "VQC"), ("rf", "RF"), ("xgb", "XGB"), ("threshold", "Threshold")]


def heldout_sub(sweep_path, already_heldout=False):
    s = np.load(sweep_path, allow_pickle=False)
    grid = s["jitters"] if "jitters" in s.files else s["values"]
    if already_heldout:
        idx = list(range(s["features"].shape[0]))
    else:
        _, idx = split_instances(s, 0.67)
    seeds = [int(x) for x in s["inst_seed"][idx]]
    return ({"features": s["features"][idx], "functional": s["functional"][idx],
             "got": s["got"][idx], "jitters": grid}, seeds)


def evasion_rate(det, sub):
    """(rate, n_evaded, n, median first-evading grid value or None). The median is the evasion COST on
    that axis; a value at the grid ceiling means the rate is a lower bound, not a converged figure."""
    ev, mj, _ = min_evading_jitter(det, sub, THRESH)
    med = float(np.median(mj[ev])) if ev.any() else None
    return float(ev.mean()), int(ev.sum()), int(len(ev)), med


def _clean_test():
    """The clean test split every detector was scored on (same split/seed as training)."""
    import yaml
    from features.dataset import load_dataset, train_val_test_split
    cfg = yaml.safe_load(open("configs/mlp.yaml"))
    X, y, _, class_names = load_dataset("data/dataset.npz")
    _, _, (Xte, yte) = train_val_test_split(X, y, val_size=cfg["val_size"],
                                            test_size=cfg["test_size"], seed=cfg["seed"])
    return Xte, yte == class_names.index("benign")


def clean_cost(result_dir, det, test):
    """(test accuracy, benign FPR) from meta.json; undefended MLP/VQC metas carry no benign_fpr, so it
    is computed on the same clean test split in that case."""
    m = json.loads((Path("results") / result_dir / "meta.json").read_text())
    fpr = m.get("benign_fpr")
    if fpr is None:
        Xte, benign = test
        fpr = float((det.prob_benign(Xte[benign]) < THRESH).mean())
    return m.get("test_accuracy"), fpr


def main():
    jitter, jh = heldout_sub("results/evasion/sweep.npz")
    size, sh = heldout_sub("results/evasion/sweep_sizeaxis_full.npz")
    comb, ch = heldout_sub("results/evasion/sweep_combined_heldout.npz", already_heldout=True)
    print(f"held-out instances: jitter {jh}\n                    size   {sh}\n                    comb   {ch}\n")

    axes = [("jitter", jitter), ("size", size), ("combined", comb)]
    test = _clean_test()
    out = {}
    hdr = f"{'detector':14s}{'variant':16s}{'clean_acc':>10s}{'FPR':>8s}" + "".join(f"{a:>12s}" for a, _ in axes)
    print(hdr)
    print("-" * len(hdr))
    for base, lbl in BASES:
        for variant, suffix in [("undefended", ""), ("single-axis", "_hardened"), ("joint", "_joint_hardened")]:
            rdir = base + suffix
            det = Detector.from_result(f"results/{rdir}")
            acc, fpr = clean_cost(rdir, det, test)
            row = {"result_dir": rdir, "clean_acc": acc, "benign_fpr": fpr}
            cells = ""
            for aname, sub in axes:
                r, k, n, med = evasion_rate(det, sub)
                row[f"evasion_{aname}"] = {"rate": r, "n_evaded": k, "n": n, "median_first_evading": med}
                medtxt = f"@{med:g}" if med is not None else ""
                cells += f"{r:>6.2f}{medtxt:<6s}"
            out.setdefault(base, {})[variant] = row
            accs = f"{acc:.3f}" if acc is not None else "  -  "
            fprs = f"{fpr:.3f}" if fpr is not None else "  -  "
            print(f"{lbl:14s}{variant:16s}{accs:>10s}{fprs:>8s}{cells}")
        print()

    Path("results/evasion/joint_harden_eval.json").write_text(json.dumps(
        {"threshold": THRESH, "held_out": {"jitter": [int(i) for i in jh], "size": [int(i) for i in sh],
         "combined": [int(i) for i in ch]}, "results": out}, indent=2))
    print("saved -> results/evasion/joint_harden_eval.json")


if __name__ == "__main__":
    main()
