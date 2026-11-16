"""Joint timing+size adversarial-training defense round for C2. The single-axis round
(evasion/harden.py) folds in only jitter-swept functional beacons, and the size axis then re-breaks it
(evasion/sweep.py --axis size_jitter). This round folds in BOTH the jitter-swept AND the size-swept
functional beacons — from the SAME train instances and split as the single-axis round — retrains every
detector into results/*_joint_hardened, and reports the hardening cost.

The paper's hypothesis: closing both realizable axes at once forces the decision boundary further into
the benign cloud (a jittered, size-padded beacon is almost identical to a benign heartbeat on both
axes), so joint hardening should raise the benign false-positive rate relative to single-axis hardening.
We measure clean accuracy and benign FPR explicitly so that cost is visible. Runs in the .venv (no
captures, no root):

    .venv/bin/python -m evasion.joint_harden

Range-wide hardening is the same round with the size line replaced by the random-box sweep
(evasion/sweep.py --box), which samples timing and size jitter together across the whole region. It
folds in the same number of extra beacons per train instance (8), spread over the region instead of
along one line:

    .venv/bin/python -m evasion.joint_harden --extra-sweep results/evasion/sweep_box.npz --suffix box_hardened
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from features.dataset import load_dataset, train_val_test_split
from evasion.harden import adversarial_flows, split_instances, train_baselines, train_mlp, train_vqc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/sweep.npz", help="jitter-axis sweep")
    p.add_argument("--size-sweep", "--extra-sweep", dest="size_sweep",
                   default="results/evasion/sweep_sizeaxis_full.npz",
                   help="second sweep over ALL instances folded in with the jitter sweep: the size line at "
                        "timing jitter 0.5 (default), or the random-box sweep for range-wide hardening")
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--train-frac", type=float, default=0.67,
                   help="must match evasion.harden so the held-out split is identical for a clean comparison")
    p.add_argument("--suffix", default="joint_hardened", help="result-dir suffix for this round")
    p.add_argument("--mlp-config", default="configs/mlp.yaml")
    p.add_argument("--vqc-config", default="configs/vqc.yaml")
    p.add_argument("--baselines-config", default="configs/baselines.yaml")
    p.add_argument("--no-baselines", action="store_true", help="skip retraining RF / XGBoost / threshold")
    args = p.parse_args()

    mlp_cfg = yaml.safe_load(open(args.mlp_config))
    vqc_cfg = yaml.safe_load(open(args.vqc_config))
    X, y, _, class_names = load_dataset(args.data)
    c2_idx = class_names.index("c2")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train_val_test_split(
        X, y, val_size=mlp_cfg["val_size"], test_size=mlp_cfg["test_size"], seed=mlp_cfg["seed"])

    jsweep = np.load(args.sweep)
    ssweep = np.load(args.size_sweep)
    jtrain, jheld = split_instances(jsweep, args.train_frac)
    strain, sheld = split_instances(ssweep, args.train_frac)
    Xj = adversarial_flows(jsweep, jtrain)       # functional timing-jittered beacons (train instances)
    Xs = adversarial_flows(ssweep, strain)       # functional size-jittered beacons (train instances)
    Xadv = np.vstack([Xj, Xs])
    yadv = np.full(len(Xadv), c2_idx)
    Xtr_aug = np.vstack([Xtr, Xadv])
    ytr_aug = np.concatenate([ytr, yadv])
    print(f"clean train {len(Xtr)} + {len(Xj)} jitter-swept + {len(Xs)} beacons from {args.size_sweep} "
          f"= {len(Xtr_aug)} train flows\n"
          f"  jitter train instances {jtrain}\n  size   train instances {strain}\n"
          f"  held out (jitter {jheld}, size {sheld}); clean test unchanged ({len(Xte)})\n")

    aug = ((Xtr_aug, ytr_aug), (Xva, yva), (Xte, yte))
    print(f"retraining {args.suffix} on augmented data (clean-test accuracy | benign false-positive rate):")
    for name, (acc, fpr) in [
        (f"classical_native_{args.suffix}",
         train_mlp(aug, None, mlp_cfg, class_names, Path(f"results/classical_native_{args.suffix}"))),
        (f"classical_matched_{args.suffix}",
         train_mlp(aug, vqc_cfg["n_qubits"], mlp_cfg, class_names, Path(f"results/classical_matched_{args.suffix}"))),
        (f"quantum_{args.suffix}",
         train_vqc(aug, vqc_cfg, class_names, Path(f"results/quantum_{args.suffix}"))),
        *train_baselines(aug, args, class_names, prefix="", suffix=args.suffix),
    ]:
        print(f"  {name:34s} acc={acc:.4f}  benign_FPR={fpr:.4f}")


if __name__ == "__main__":
    main()
