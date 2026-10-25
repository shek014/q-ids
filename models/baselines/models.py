"""Reference detectors the MLP/VQC are compared against: the models a flow-level NIDS would actually
deploy (Random Forest, gradient-boosted trees) and a single-feature statistical rule of the kind
beacon hunters use (low inter-arrival-time CV => beacon; high throughput => bulk exfil).

These deliberately use library implementations. The "no framework" rule in the README applies to the
MLP-vs-VQC comparison (both written out so neither gets a better-optimized engine); baselines exist to
show where that comparison sits relative to standard practice, so they should be the standard thing.

Every model exposes predict_proba(X) -> (n, n_classes), predict(X), param_count(), save(), load(), so
evasion/detector.py can wrap them exactly like the MLP and VQC.
"""
import json
import pickle

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class ThresholdDetector:
    """One feature, one cut. Flags the attack class when the (standardized) feature is on the attack
    side of `threshold`. The direction and cut are fit on train to maximize balanced accuracy, so it is
    the best possible single-threshold rule on that feature. predict_proba is a logistic of the signed
    margin (sharpness k): exactly the rule at P=0.5, but graded, so it can be operated at any FPR."""

    def __init__(self, feature_idx, feature_name, sharpness=4.0, attack_idx=1, n_classes=2):
        self.feature_idx = int(feature_idx)
        self.feature_name = feature_name
        self.sharpness = float(sharpness)
        self.attack_idx = int(attack_idx)
        self.n_classes = int(n_classes)
        self.threshold = None
        self.attack_below = None   # True: attack if x < threshold (e.g. low cv_iat beacon)

    def fit(self, X, y):
        x = X[:, self.feature_idx]
        is_att = y == self.attack_idx
        u = np.unique(x)
        cands = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
        best = (-1.0, None, None)
        for below in (True, False):
            for t in cands:
                pred = (x < t) if below else (x > t)
                tpr = pred[is_att].mean() if is_att.any() else 0.0
                tnr = (~pred[~is_att]).mean() if (~is_att).any() else 0.0
                bal = 0.5 * (tpr + tnr)
                if bal > best[0]:
                    best = (bal, float(t), below)
        _, self.threshold, self.attack_below = best
        return self

    def predict_proba(self, X):
        x = np.atleast_2d(X)[:, self.feature_idx]
        margin = (self.threshold - x) if self.attack_below else (x - self.threshold)
        p_att = _sigmoid(self.sharpness * margin)
        out = np.zeros((len(x), self.n_classes))
        out[:, self.attack_idx] = p_att
        out[:, 1 - self.attack_idx] = 1.0 - p_att
        return out

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def param_count(self):
        return 2   # threshold + direction

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"feature_idx": self.feature_idx, "feature_name": self.feature_name,
                       "sharpness": self.sharpness, "attack_idx": self.attack_idx,
                       "n_classes": self.n_classes, "threshold": self.threshold,
                       "attack_below": self.attack_below}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(d["feature_idx"], d["feature_name"], d["sharpness"], d["attack_idx"], d["n_classes"])
        m.threshold, m.attack_below = d["threshold"], d["attack_below"]
        return m


class TreeModel:
    """Thin wrapper over a fitted sklearn RandomForestClassifier or xgboost XGBClassifier, giving it the
    same save/load/param_count surface as the MLP and VQC."""

    def __init__(self, est, kind):
        self.est = est
        self.kind = kind

    def predict_proba(self, X):
        return self.est.predict_proba(np.atleast_2d(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def param_count(self):
        """Total tree nodes — the closest analogue of a parameter count for an ensemble."""
        if self.kind == "random_forest":
            return int(sum(t.tree_.node_count for t in self.est.estimators_))
        df = self.est.get_booster().trees_to_dataframe()
        return int(len(df))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"kind": self.kind, "est": self.est}, f)

    @classmethod
    def load(cls, path):
        # only ever loads files this pipeline wrote into results/ — never an untrusted pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["est"], d["kind"])
