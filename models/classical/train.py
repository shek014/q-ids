import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from features.dataset import load_dataset, preprocess, train_val_test_split
from models.classical.mlp import MLP


def balanced_class_weight(y):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    return {int(c): n / (len(classes) * count) for c, count in zip(classes, counts)}


def run(data_path, config_path, out_dir, pca_components=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    X, y, feature_names, class_names = load_dataset(data_path)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, val_size=cfg["val_size"], test_size=cfg["test_size"], seed=cfg["seed"]
    )

    # native input = all raw features; matched input = the same PCA components the VQC sees.
    (X_train, X_val, X_test), pre = preprocess(
        (X_train, X_val, X_test), n_components=pca_components
    )
    input_mode = "matched" if pca_components is not None else "native"

    class_weight = balanced_class_weight(y_train) if cfg.get("class_weight") == "balanced" else None

    layer_sizes = [X_train.shape[1], *cfg["hidden_sizes"], len(class_names)]
    model = MLP(layer_sizes, optimizer=cfg["optimizer"], learning_rate=cfg["learning_rate"], seed=cfg["seed"])

    start = time.perf_counter()
    history = model.fit(
        X_train, y_train, X_val=X_val, y_val=y_val,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        class_weight=class_weight, patience=cfg["patience"], seed=cfg["seed"],
    )
    train_time = time.perf_counter() - start

    y_proba = model.predict_proba(X_test)
    test_acc = float(np.mean(np.argmax(y_proba, axis=1) == y_test))
    print(f"[{input_mode}] test accuracy: {test_acc:.4f}  |  params: {model.param_count()}  "
          f"|  features: {X_train.shape[1]}  |  train time: {train_time:.2f}s")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.npz")
    np.savez(out / "history.npz", **{k: np.array(v) for k, v in history.items()})
    np.savez(out / "test_predictions.npz", y_true=y_test, y_proba=y_proba)
    (out / "meta.json").write_text(json.dumps({
        "model": "classical_mlp",
        "input_mode": input_mode,
        "param_count": model.param_count(),
        "train_time_seconds": train_time,
        "test_accuracy": test_acc,
        "n_features": X_train.shape[1],
        "pca_components": pca_components,
        "pca_explained_variance_ratio": pre["explained_variance_ratio"],
        "feature_names": feature_names,
        "class_names": class_names,
        "config": cfg,
    }, indent=2))
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.npz")
    parser.add_argument("--config", default="configs/mlp.yaml")
    parser.add_argument("--out", default="results/classical")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="reduce to N PCA components (matched-input regime, e.g. the VQC's "
                             "qubit count); omit to train on all raw features (native input)")
    args = parser.parse_args()
    run(args.data, args.config, args.out, pca_components=args.pca_components)


if __name__ == "__main__":
    main()
