"""A thin wrapper that reloads a trained detector together with the exact preprocessing it was
trained under, and scores a raw 17-feature flow -> P(benign). The evasion search asks one question
of it per query: does the detector call this beacon benign?

Preprocessing is reconstructed by re-deriving the train split (same seed/sizes from the model's
saved config) and re-fitting the scaler/PCA on it — deterministic and, run in the .venv, bit-for-bit
what training used. The native MLP needs only the scaler (plain z-score), so it is fully robust; the
PCA/angle path (matched MLP, VQC) is wired but the VQC's angle reconstruction is deferred to the
transferability chunk.
"""
import json
from pathlib import Path

import numpy as np

from features.dataset import load_dataset, train_val_test_split, preprocess
from models.classical.mlp import MLP


class Detector:
    def __init__(self, model, info, benign_idx, meta, name):
        self.model = model
        self.info = info
        self.benign_idx = benign_idx
        self.meta = meta
        self.name = name

    @classmethod
    def from_result(cls, result_dir, data_path="data/dataset.npz"):
        result_dir = Path(result_dir)
        meta = json.loads((result_dir / "meta.json").read_text())
        cfg = meta["config"]

        X, y, _, class_names = load_dataset(data_path)
        (X_train, _), (X_val, _), (X_test, _) = train_val_test_split(
            X, y, val_size=cfg["val_size"], test_size=cfg["test_size"], seed=cfg["seed"])

        to_angles = meta["model"] == "vqc"
        _, info = preprocess((X_train, X_val, X_test),
                             n_components=meta.get("pca_components"), to_angles=to_angles)

        if meta["model"] == "classical_mlp":
            model = MLP.load(str(result_dir / "model.npz"))
        else:
            raise NotImplementedError(
                f"detector for model '{meta['model']}' not built yet (VQC lands with transferability)")

        return cls(model, info, class_names.index("benign"), meta, result_dir.name)

    def transform(self, raw):
        Z = self.info["scaler"].transform(np.atleast_2d(raw))
        if self.info["pca"] is not None:
            Z = self.info["pca"].transform(Z)
        if self.info.get("angle_scaled"):
            raise NotImplementedError("angle-scaling reconstruction is deferred (VQC transferability)")
        return Z

    def prob_benign(self, raw):
        """raw: (17,) or (n, 17). Returns P(benign) as a length-n array."""
        return self.model.predict_proba(self.transform(raw))[:, self.benign_idx]

    def calls_benign(self, raw, threshold=0.5):
        return bool(self.prob_benign(raw)[0] >= threshold)
