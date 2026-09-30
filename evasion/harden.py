"""Adversarial-training defense round. Fold evaded beacons (from the capture-once sweep) into each
detector's training set as c2, retrain, and report the hardening cost. The interesting tension: a
jittered beacon is almost identical to a benign heartbeat, so teaching the detector "jittered beacon
= c2" risks it flagging real benign heartbeats — a false-positive cost we measure explicitly.

To avoid leakage, only sweep seeds in --adv-seeds go into training; the rest are held out for
evasion evaluation (evasion/harden_eval.py). Retrains all three models (MLP native, MLP matched,
VQC) into <name>_hardened/ result dirs, each with a saved prep.npz so the detector scores correctly
despite being fit on augmented data. Runs in the .venv (no captures here):

    .venv/bin/python -m evasion.harden --sweep results/evasion/sweep.npz --adv-seeds 0-9
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from features.dataset import load_dataset, preprocess, train_val_test_split, FEATURE_NAMES
from models.classical.mlp import MLP
from models.classical.train import balanced_class_weight


def parse_seeds(spec):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def adversarial_flows(sweep, seeds):
    """All captured, functional beacon flows from the given seeds — jittered beacons that a hardened
    detector should still recognize as c2."""
    feats, got, func = sweep["features"], sweep["got"], sweep["functional"]
    rows = [feats[si, ji] for si in seeds for ji in range(feats.shape[1])
            if got[si, ji] and func[si, ji]]
    return np.array(rows)


def _save_prep(out, info, X_train, to_angles):
    d = {"scaler_mean": info["scaler"].mean_, "scaler_std": info["scaler"].std_}
    if info["pca"] is not None:
        d["pca_mean"], d["pca_comp"] = info["pca"].mean_, info["pca"].components_
    if to_angles:
        Z = info["scaler"].transform(X_train)
        if info["pca"] is not None:
            Z = info["pca"].transform(Z)
        d["angle_ref_min"] = Z.min(axis=0)
        d["angle_span"] = np.clip(Z.max(axis=0) - Z.min(axis=0), 1e-8, None)
        d["angle_bound"] = np.pi
    np.savez(out / "prep.npz", **d)


def _benign_fpr(model, Xte_t, yte, benign_idx=0, c2_idx=1):
    pred = model.predict(Xte_t)
    benign = yte == benign_idx
    return float((pred[benign] == c2_idx).mean()) if benign.any() else 0.0


def _write_meta(out, model_name, input_mode, pca, acc, fpr, class_names, cfg, n_feat, extra=None):
    meta = {"model": model_name, "input_mode": input_mode, "pca_components": pca,
            "test_accuracy": acc, "benign_fpr": fpr, "n_features": n_feat,
            "feature_names": FEATURE_NAMES, "class_names": class_names, "config": cfg,
            "adversarially_trained": True}
    meta.update(extra or {})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def train_mlp(splits, pca, cfg, class_names, out):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    (Xtr_t, Xva_t, Xte_t), info = preprocess((Xtr, Xva, Xte), n_components=pca)
    cw = balanced_class_weight(ytr) if cfg.get("class_weight") == "balanced" else None
    layers = [Xtr_t.shape[1], *cfg["hidden_sizes"], len(class_names)]
    m = MLP(layers, optimizer=cfg["optimizer"], learning_rate=cfg["learning_rate"], seed=cfg["seed"])
    m.fit(Xtr_t, ytr, X_val=Xva_t, y_val=yva, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
          class_weight=cw, patience=cfg["patience"], seed=cfg["seed"])
    acc = float((m.predict(Xte_t) == yte).mean())
    fpr = _benign_fpr(m, Xte_t, yte)
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "model.npz")
    _save_prep(out, info, Xtr, to_angles=False)
    _write_meta(out, "classical_mlp", "matched" if pca else "native", pca, acc, fpr,
                class_names, cfg, Xtr_t.shape[1])
    return acc, fpr


def train_vqc(splits, cfg, class_names, out):
    from models.quantum.vqc import VQC
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    nq = cfg["n_qubits"]
    (Xtr_t, Xva_t, Xte_t), info = preprocess((Xtr, Xva, Xte), n_components=nq, to_angles=True)
    m = VQC(n_qubits=nq, n_classes=len(class_names), n_layers=cfg["n_layers"],
            learning_rate=cfg["learning_rate"], seed=cfg["seed"])
    m.fit(Xtr_t, ytr, X_val=Xva_t, y_val=yva, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
          patience=cfg["patience"], seed=cfg["seed"])
    acc = float((m.predict(Xte_t) == yte).mean())
    fpr = _benign_fpr(m, Xte_t, yte)
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "model.npz")
    _save_prep(out, info, Xtr, to_angles=True)
    _write_meta(out, "quantum_vqc", "matched", nq, acc, fpr, class_names, cfg, Xtr_t.shape[1],
                extra={"n_qubits": nq})
    return acc, fpr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/sweep.npz")
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--adv-seeds", default="0-9", help="sweep seeds folded into training (rest held out)")
    p.add_argument("--mlp-config", default="configs/mlp.yaml")
    p.add_argument("--vqc-config", default="configs/vqc.yaml")
    args = p.parse_args()

    mlp_cfg = yaml.safe_load(open(args.mlp_config))
    vqc_cfg = yaml.safe_load(open(args.vqc_config))
    X, y, _, class_names = load_dataset(args.data)
    c2_idx = class_names.index("c2")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train_val_test_split(
        X, y, val_size=mlp_cfg["val_size"], test_size=mlp_cfg["test_size"], seed=mlp_cfg["seed"])

    sweep = np.load(args.sweep)
    adv_seeds = parse_seeds(args.adv_seeds)
    Xadv = adversarial_flows(sweep, adv_seeds)
    yadv = np.full(len(Xadv), c2_idx)
    Xtr_aug = np.vstack([Xtr, Xadv])
    ytr_aug = np.concatenate([ytr, yadv])
    print(f"clean train {len(Xtr)} + {len(Xadv)} adversarial beacons (seeds {args.adv_seeds}) "
          f"= {len(Xtr_aug)} train flows; clean test unchanged ({len(Xte)})\n")

    aug = ((Xtr_aug, ytr_aug), (Xva, yva), (Xte, yte))
    print("retraining on augmented data (clean-test accuracy | benign false-positive rate):")
    for name, (acc, fpr) in [
        ("classical_native_hardened", train_mlp(aug, None, mlp_cfg, class_names, Path("results/classical_native_hardened"))),
        ("classical_matched_hardened", train_mlp(aug, vqc_cfg["n_qubits"], mlp_cfg, class_names, Path("results/classical_matched_hardened"))),
        ("quantum_hardened", train_vqc(aug, vqc_cfg, class_names, Path("results/quantum_hardened"))),
    ]:
        print(f"  {name:28s} acc={acc:.4f}  benign_FPR={fpr:.4f}")


if __name__ == "__main__":
    main()
