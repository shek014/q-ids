"""A thin wrapper that reloads a trained detector together with the exact preprocessing it was
trained under, and scores a raw 17-feature flow -> P(benign). The evasion search asks one question
of it per query: does the detector call this beacon benign?

Preprocessing is taken from a saved prep.npz in the result dir when present (adversarially-trained
models are fit on augmented data, so their transform can't be re-derived from the original split);
otherwise it is reconstructed by re-deriving the train split and re-fitting — deterministic and, in
the .venv, bit-for-bit what training used. Both paths reduce to the same arrays (scaler, optional
PCA, optional AngleEmbedding scaling) applied in transform().
"""
import json
from pathlib import Path

import numpy as np

from features.dataset import load_dataset, train_val_test_split, preprocess
from models.classical.mlp import MLP


class Detector:
    def __init__(self, model, prep, benign_idx, meta, name):
        self.model = model
        self.prep = prep            # {scaler_mean, scaler_std, pca_mean|None, pca_comp|None, angle|None}
        self.benign_idx = benign_idx
        self.meta = meta
        self.name = name

    @classmethod
    def from_result(cls, result_dir, data_path="data/dataset.npz"):
        result_dir = Path(result_dir)
        meta = json.loads((result_dir / "meta.json").read_text())

        if meta["model"] == "classical_mlp":
            model = MLP.load(str(result_dir / "model.npz"))
        elif meta["model"] == "quantum_vqc":
            from models.quantum.vqc import VQC   # lazy: only pull in pennylane for a VQC detector
            model = VQC.load(str(result_dir / "model.npz"))
        else:
            raise NotImplementedError(f"detector for model '{meta['model']}' not supported")

        prep_path = result_dir / "prep.npz"
        prep = cls._prep_from_arrays(np.load(prep_path)) if prep_path.exists() \
            else cls._refit_prep(meta, data_path)
        return cls(model, prep, meta["class_names"].index("benign"), meta, result_dir.name)

    @staticmethod
    def _refit_prep(meta, data_path):
        cfg = meta["config"]
        X, y, _, _ = load_dataset(data_path)
        (X_train, _), (X_val, _), (X_test, _) = train_val_test_split(
            X, y, val_size=cfg["val_size"], test_size=cfg["test_size"], seed=cfg["seed"])
        to_angles = meta["model"] == "quantum_vqc"
        _, info = preprocess((X_train, X_val, X_test),
                             n_components=meta.get("pca_components"), to_angles=to_angles)
        prep = {"scaler_mean": info["scaler"].mean_, "scaler_std": info["scaler"].std_,
                "pca_mean": None, "pca_comp": None, "angle": None}
        if info["pca"] is not None:
            prep["pca_mean"], prep["pca_comp"] = info["pca"].mean_, info["pca"].components_
        if to_angles:
            Z = info["scaler"].transform(X_train)
            if info["pca"] is not None:
                Z = info["pca"].transform(Z)
            prep["angle"] = {"ref_min": Z.min(axis=0),
                             "span": np.clip(Z.max(axis=0) - Z.min(axis=0), 1e-8, None), "bound": np.pi}
        return prep

    @staticmethod
    def _prep_from_arrays(d):
        prep = {"scaler_mean": d["scaler_mean"], "scaler_std": d["scaler_std"],
                "pca_mean": None, "pca_comp": None, "angle": None}
        if "pca_comp" in d and d["pca_comp"].size:
            prep["pca_mean"], prep["pca_comp"] = d["pca_mean"], d["pca_comp"]
        if "angle_ref_min" in d and d["angle_ref_min"].size:
            prep["angle"] = {"ref_min": d["angle_ref_min"], "span": d["angle_span"],
                             "bound": float(d["angle_bound"])}
        return prep

    def transform(self, raw):
        Z = (np.atleast_2d(raw) - self.prep["scaler_mean"]) / self.prep["scaler_std"]
        if self.prep["pca_comp"] is not None:
            Z = (Z - self.prep["pca_mean"]) @ self.prep["pca_comp"].T
        a = self.prep["angle"]
        if a is not None:
            Z = (Z - a["ref_min"]) / a["span"] * (2 * a["bound"]) - a["bound"]
        return Z

    def prob_benign(self, raw):
        """raw: (17,) or (n, 17). Returns P(benign) as a length-n array."""
        return self.model.predict_proba(self.transform(raw))[:, self.benign_idx]

    def calls_benign(self, raw, threshold=0.5):
        return bool(self.prob_benign(raw)[0] >= threshold)
