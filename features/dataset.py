import numpy as np

FEATURE_NAMES = [
    "duration", "packet_count", "byte_count", "mean_packet_size", "std_packet_size",
    "packets_per_second", "bytes_per_second", "syn_count", "ack_count", "fin_count",
    "rst_count", "mean_iat", "std_iat", "unique_dst_ports", "unique_src_ports", "protocol",
]

CLASS_NAMES = ["benign", "dos", "recon", "spoof"]


def save_dataset(path, X, y, feature_names=FEATURE_NAMES, class_names=CLASS_NAMES):
    np.savez(path, X=X, y=y, feature_names=np.array(feature_names), class_names=np.array(class_names))


def load_dataset(path):
    data = np.load(path, allow_pickle=False)
    return data["X"], data["y"], data["feature_names"].tolist(), data["class_names"].tolist()


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=None):
    """Stratified split: each class is shuffled and split independently, then recombined,
    so the class balance in every split matches the source data."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(n * val_size))
        n_test = int(round(n * test_size))
        val_idx.append(idx[:n_val])
        test_idx.append(idx[n_val:n_val + n_test])
        train_idx.append(idx[n_val + n_test:])

    train_idx = rng.permutation(np.concatenate(train_idx))
    val_idx = rng.permutation(np.concatenate(val_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))

    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


class StandardScaler:
    """z-score scaling, fit on train only and reused for val/test — no sklearn dependency."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCA:
    """Dimensionality reduction via SVD — used to fit flow features into a qubit budget
    for the VQC (see README > The dimensionality constraint). Not used by the classical model."""

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[:self.n_components]
        total_var = np.sum(s ** 2)
        self.explained_variance_ratio_ = (s[:self.n_components] ** 2) / total_var
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def preprocess(splits, n_components=None, to_angles=False, angle_bound=np.pi):
    """Shared feature preprocessing for both trainers, so the classical and quantum models
    can be run on identical inputs (see README > The dimensionality constraint).

    splits is (X_train, X_val, X_test). Every transform is fit on the train split only and
    reused for val/test, so no information leaks across the split boundary:
      1. standardize (z-score);
      2. if n_components is set, reduce to that many PCA components;
      3. if to_angles, rescale each feature into [-angle_bound, angle_bound] — required by the
         VQC's AngleEmbedding, a monotonic per-feature rescale that carries no extra information
         (so an MLP on the PCA components and a VQC on the angle-scaled PCA components see the
         same information — the matched-input comparison).

    Returns (transformed splits, info) where info carries the fitted scaler/pca and, when PCA
    is used, the explained-variance ratio.
    """
    X_train, X_val, X_test = splits
    scaler = StandardScaler().fit(X_train)
    parts = [scaler.transform(X) for X in (X_train, X_val, X_test)]

    info = {"scaler": scaler, "pca": None, "n_components": n_components,
            "explained_variance_ratio": None, "angle_scaled": to_angles}

    if n_components is not None:
        pca = PCA(n_components).fit(parts[0])
        parts = [pca.transform(X) for X in parts]
        info["pca"] = pca
        info["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    if to_angles:
        ref_min, ref_max = parts[0].min(axis=0), parts[0].max(axis=0)
        span = np.clip(ref_max - ref_min, 1e-8, None)
        parts = [(X - ref_min) / span * (2 * angle_bound) - angle_bound for X in parts]

    return parts, info
