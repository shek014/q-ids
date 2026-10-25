"""Train a reference detector (Random Forest, XGBoost, or a single-feature threshold rule) on the same
split as the MLP/VQC, on the native 17 features. Writes the same result-dir layout as the other
trainers (model file, prep.npz, meta.json, history.npz, test_predictions.npz), so evasion scoring and
hardening treat it like any other detector.

    python -m models.baselines.train --kind rf  --data data/dataset.npz --out results/rf
    python -m models.baselines.train --kind xgb --data data/dataset.npz --out results/xgb
    python -m models.baselines.train --kind threshold --feature cv_iat --data data/dataset.npz \
        --out results/threshold
    # exfil: same, with --data data/exfil_dataset.npz and --feature bytes_per_second for the rule
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from features.dataset import FEATURE_NAMES, load_dataset, preprocess, train_val_test_split
from models.baselines.models import ThresholdDetector, TreeModel

KIND_TO_MODEL = {"rf": "random_forest", "xgb": "xgboost", "threshold": "threshold"}
MODEL_FILE = {"random_forest": "model.pkl", "xgboost": "model.pkl", "threshold": "model.json"}


def default_feature(class_names):
    """The signal each attack family's detector keys on: timing regularity for C2, throughput for exfil."""
    return "bytes_per_second" if "exfil" in class_names else "cv_iat"


def _fit(kind, Xtr, ytr, Xva, yva, cfg, class_names, feature):
    attack_idx = 1 - class_names.index("benign")
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        c = cfg["rf"]
        est = RandomForestClassifier(n_estimators=c["n_estimators"], max_depth=c.get("max_depth"),
                                     min_samples_leaf=c.get("min_samples_leaf", 1),
                                     class_weight="balanced", random_state=cfg["seed"], n_jobs=-1)
        est.fit(Xtr, ytr)
        return TreeModel(est, kind)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        c = cfg["xgb"]
        n_pos, n_neg = int((ytr == attack_idx).sum()), int((ytr != attack_idx).sum())
        est = XGBClassifier(n_estimators=c["n_estimators"], max_depth=c["max_depth"],
                            learning_rate=c["learning_rate"], subsample=c["subsample"],
                            colsample_bytree=c["colsample_bytree"],
                            scale_pos_weight=n_neg / max(n_pos, 1) if attack_idx == 1 else n_pos / max(n_neg, 1),
                            early_stopping_rounds=c["early_stopping_rounds"], eval_metric="logloss",
                            random_state=cfg["seed"], n_jobs=-1)
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return TreeModel(est, kind)
    if kind == "threshold":
        m = ThresholdDetector(FEATURE_NAMES.index(feature), feature, cfg["threshold"]["sharpness"],
                              attack_idx=attack_idx, n_classes=len(class_names))
        return m.fit(Xtr, ytr)
    raise ValueError(kind)


def fit_baseline(kind, splits, cfg, class_names, out, feature=None, extra_meta=None):
    """kind: random_forest | xgboost | threshold. splits: ((Xtr,ytr),(Xva,yva),(Xte,yte)) of RAW
    features. Standardizes (fit on train; a monotone per-feature map, so it changes nothing for trees
    or a single cut but keeps every detector on the same prep path), fits, evaluates, writes `out`.
    Returns (test_accuracy, benign_fpr)."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits
    (Xtr_t, Xva_t, Xte_t), info = preprocess((Xtr, Xva, Xte))
    feature = feature or default_feature(class_names)

    start = time.perf_counter()
    model = _fit(kind, Xtr_t, ytr, Xva_t, yva, cfg, class_names, feature)
    train_time = time.perf_counter() - start

    benign_idx = class_names.index("benign")
    proba = model.predict_proba(Xte_t)
    pred = np.argmax(proba, axis=1)
    acc = float((pred == yte).mean())
    benign = yte == benign_idx
    fpr = float((pred[benign] != benign_idx).mean()) if benign.any() else 0.0

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / MODEL_FILE[kind])
    np.savez(out / "prep.npz", scaler_mean=info["scaler"].mean_, scaler_std=info["scaler"].std_)
    np.savez(out / "history.npz")
    np.savez(out / "test_predictions.npz", y_true=yte, y_proba=proba)
    meta = {"model": kind, "input_mode": "native", "pca_components": None,
            "param_count": model.param_count(), "train_time_seconds": train_time,
            "test_accuracy": acc, "benign_fpr": fpr, "n_features": Xtr_t.shape[1],
            "feature_names": FEATURE_NAMES, "class_names": class_names,
            "config": cfg}
    if kind == "threshold":
        meta["threshold_feature"] = feature
        meta["threshold_attack_below"] = bool(model.attack_below)
    meta.update(extra_meta or {})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return acc, fpr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", required=True, choices=list(KIND_TO_MODEL))
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--config", default="configs/baselines.yaml")
    p.add_argument("--feature", default=None,
                   help="threshold rule feature (default: cv_iat for C2, bytes_per_second for exfil)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    X, y, _, class_names = load_dataset(args.data)
    splits = train_val_test_split(X, y, val_size=cfg["val_size"], test_size=cfg["test_size"],
                                  seed=cfg["seed"])
    kind = KIND_TO_MODEL[args.kind]
    acc, fpr = fit_baseline(kind, splits, cfg, class_names, args.out, feature=args.feature)
    print(f"[{kind}] test accuracy {acc:.4f}  benign FPR {fpr:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
