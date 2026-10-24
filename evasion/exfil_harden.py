"""Adversarial-training defense round for EXFIL (PROTOTYPE). Fold evaded throttled-exfil flows (from
the capture-once throttle sweep) into each detector's training set as exfil, retrain, and report the
hardening cost. The tension mirrors C2's: a throttled exfil is almost identical to a low-rate benign
trickle, so teaching the detector "throttled exfil = exfil" risks it flagging legitimate low-rate
transfers — a false-positive cost we measure explicitly.

To avoid leakage, only sweep seeds in --adv-seeds go into training; the rest are held out for evasion
evaluation (evasion/exfil_harden_eval.py). Reuses the generic training/prep helpers from
evasion/harden.py and retrains all three models (MLP native, MLP matched, VQC) into
results/exfil_*_hardened/, each with a saved prep.npz. Runs in the .venv (no captures here):

    .venv/bin/python -m evasion.exfil_harden --sweep results/evasion/exfil_sweep.npz --adv-seeds 0-9
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from features.dataset import load_dataset, train_val_test_split
from evasion.harden import adversarial_flows, split_instances, train_mlp, train_vqc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/exfil_sweep.npz")
    p.add_argument("--data", default="data/exfil_dataset.npz")
    p.add_argument("--train-frac", type=float, default=0.67,
                   help="fraction of instances per config folded into adversarial training (rest held out)")
    p.add_argument("--mlp-config", default="configs/mlp.yaml")
    p.add_argument("--vqc-config", default="configs/vqc.yaml")
    args = p.parse_args()

    mlp_cfg = yaml.safe_load(open(args.mlp_config))
    vqc_cfg = yaml.safe_load(open(args.vqc_config))
    X, y, _, class_names = load_dataset(args.data)
    attack_idx = class_names.index("exfil")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train_val_test_split(
        X, y, val_size=mlp_cfg["val_size"], test_size=mlp_cfg["test_size"], seed=mlp_cfg["seed"])

    sweep = np.load(args.sweep)
    train_idx, heldout_idx = split_instances(sweep, args.train_frac)
    Xadv = adversarial_flows(sweep, train_idx)           # all captured, functional throttled exfil flows
    yadv = np.full(len(Xadv), attack_idx)
    Xtr_aug = np.vstack([Xtr, Xadv])
    ytr_aug = np.concatenate([ytr, yadv])
    print(f"clean train {len(Xtr)} + {len(Xadv)} adversarial throttled-exfil flows (train instances "
          f"{train_idx}, {len(heldout_idx)} held out) = {len(Xtr_aug)} train flows; clean test unchanged ({len(Xte)})\n")

    aug = ((Xtr_aug, ytr_aug), (Xva, yva), (Xte, yte))
    print("retraining on augmented data (clean-test accuracy | benign false-positive rate):")
    for name, (acc, fpr) in [
        ("exfil_classical_native_hardened",
         train_mlp(aug, None, mlp_cfg, class_names, Path("results/exfil_classical_native_hardened"))),
        ("exfil_classical_matched_hardened",
         train_mlp(aug, vqc_cfg["n_qubits"], mlp_cfg, class_names, Path("results/exfil_classical_matched_hardened"))),
        ("exfil_quantum_hardened",
         train_vqc(aug, vqc_cfg, class_names, Path("results/exfil_quantum_hardened"))),
    ]:
        print(f"  {name:34s} acc={acc:.4f}  benign_FPR={fpr:.4f}")


if __name__ == "__main__":
    main()
