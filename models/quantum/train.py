import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from features.dataset import load_dataset, preprocess, train_val_test_split
from models.quantum.vqc import VQC


def run(data_path, config_path, out_dir):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    X, y, feature_names, class_names = load_dataset(data_path)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, val_size=cfg["val_size"], test_size=cfg["test_size"], seed=cfg["seed"]
    )

    # standardize -> PCA down to the qubit budget -> rescale into angle range for AngleEmbedding.
    n_qubits = cfg["n_qubits"]
    (X_train, X_val, X_test), pre = preprocess(
        (X_train, X_val, X_test), n_components=n_qubits, to_angles=True
    )

    model = VQC(n_qubits=n_qubits, n_classes=len(class_names), n_layers=cfg["n_layers"],
                learning_rate=cfg["learning_rate"], seed=cfg["seed"])

    start = time.perf_counter()
    history = model.fit(
        X_train, y_train, X_val=X_val, y_val=y_val,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        patience=cfg["patience"], seed=cfg["seed"],
    )
    train_time = time.perf_counter() - start

    y_proba = model.predict_proba(X_test)
    test_acc = float(np.mean(np.argmax(y_proba, axis=1) == y_test))
    print(f"test accuracy: {test_acc:.4f}  |  params: {model.param_count()}  |  train time: {train_time:.2f}s")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.npz")
    np.savez(out / "history.npz", **{k: np.array(v) for k, v in history.items()})
    np.savez(out / "test_predictions.npz", y_true=y_test, y_proba=y_proba)
    (out / "meta.json").write_text(json.dumps({
        "model": "quantum_vqc",
        "input_mode": "matched",
        "param_count": model.param_count(),
        "train_time_seconds": train_time,
        "test_accuracy": test_acc,
        "n_features": X_train.shape[1],
        "n_qubits": n_qubits,
        "pca_components": n_qubits,
        "pca_explained_variance_ratio": pre["explained_variance_ratio"],
        "feature_names": feature_names,
        "class_names": class_names,
        "config": cfg,
    }, indent=2))
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.npz")
    parser.add_argument("--config", default="configs/vqc.yaml")
    parser.add_argument("--out", default="results/quantum")
    args = parser.parse_args()
    run(args.data, args.config, args.out)


if __name__ == "__main__":
    main()
