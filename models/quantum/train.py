import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from features.dataset import PCA, StandardScaler, load_dataset, train_val_test_split
from models.quantum.vqc import VQC


def scale_to_angles(X, ref_min, ref_max, bound=np.pi):
    """Map PCA-reduced features into [-bound, bound] so they behave as sane rotation
    angles for AngleEmbedding — fit range comes from the train split only."""
    span = np.clip(ref_max - ref_min, 1e-8, None)
    return (X - ref_min) / span * (2 * bound) - bound


def run(data_path, config_path, out_dir):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    X, y, feature_names, class_names = load_dataset(data_path)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, val_size=cfg["val_size"], test_size=cfg["test_size"], seed=cfg["seed"]
    )

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    n_qubits = cfg["n_qubits"]
    pca = PCA(n_components=n_qubits).fit(X_train)
    X_train, X_val, X_test = pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)

    ref_min, ref_max = X_train.min(axis=0), X_train.max(axis=0)
    X_train = scale_to_angles(X_train, ref_min, ref_max)
    X_val = scale_to_angles(X_val, ref_min, ref_max)
    X_test = scale_to_angles(X_test, ref_min, ref_max)

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
        "param_count": model.param_count(),
        "train_time_seconds": train_time,
        "test_accuracy": test_acc,
        "n_qubits": n_qubits,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
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
