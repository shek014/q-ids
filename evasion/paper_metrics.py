"""Scoring conventions used by the paper's tables, shared by paper/make_tables.py and
evasion/seed_variance.py.

An instance counts toward a detector's evasion rate only if the detector flags the unmodified attack
(the grid floor). Among those, the instance is evaded at the first grid point where the flow is
functional and P(benign) >= threshold. Minimal evading values are reported as a bracket
(last captured non-evading value, first evading value], medians taken separately over each end.
"""
import numpy as np

from features.dataset import load_dataset, train_val_test_split

THRESHOLD = 0.5


def grid_of(sweep):
    for key in ("values", "jitters", "throttles"):
        if key in sweep:
            return np.asarray(sweep[key], float)
    raise KeyError("sweep has no grid")


def wilson(k, n, z=1.96):
    """Wilson 95% interval for k successes in n trials; (nan, nan) when n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def score(det, sweep, idx=None, threshold=THRESHOLD):
    """Score a capture-once sweep against one detector. Returns a dict of per-instance arrays:
    detected (flagged at the grid floor), evaded (detected and evaded), first (index of the first
    evading grid point, -1 if none), P (P(benign) per flow, nan where no flow was captured)."""
    F, got, fn = sweep["features"], sweep["got"], sweep["functional"]
    if idx is not None:
        F, got, fn = F[idx], got[idx], fn[idx]
    P = det.prob_benign_grid(F, got)
    ok = got & fn & (P >= threshold)
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    detected = got[:, 0] & (P[:, 0] < threshold)
    return {"detected": detected, "evaded": detected & (first >= 0), "first": first, "P": P,
            "got": got, "grid": grid_of(sweep)}


def summary(r, completion_time=None):
    """Table row for one detector on one sweep (detected-only)."""
    det, ev, first, grid, got = r["detected"], r["evaded"], r["first"], r["grid"], r["got"]
    k, n = int(ev.sum()), int(det.sum())
    row = {"naive_detected": [n, len(det)], "evaded": [k, n], "wilson": wilson(k, n),
           "bracket": None, "median_min": None, "max_min": None, "median_queries": None,
           "n_above": None, "median_time": None}
    if k:
        idx = np.where(ev)[0]
        hi = grid[first[idx]]
        lo = []
        for i in idx:
            prev = [grid[j] for j in range(first[i]) if got[i, j]]
            lo.append(prev[-1] if prev else 0.0)
        row.update(bracket=[float(np.median(lo)), float(np.median(hi))],
                   median_min=float(np.median(hi)), max_min=float(hi.max()),
                   median_queries=float(np.median(first[idx] + 1)))
        if completion_time is not None:
            row["median_time"] = float(np.median([completion_time[i, first[i]] for i in idx]))
    return row


def paired(a, b):
    """Instances detected and evaded by both: counts of b needing a higher, lower, equal grid index."""
    both = a["evaded"] & b["evaded"]
    fa, fb = a["first"][both], b["first"][both]
    return {"n": int(both.sum()), "b_higher": int((fb > fa).sum()), "b_lower": int((fb < fa).sum()),
            "tie": int((fb == fa).sum())}


def transfer(src, dst, threshold=THRESHOLD):
    """Evasions tuned against src (its first evading flow), scored on dst, over instances dst detected."""
    m = src["evaded"] & dst["detected"]
    k = int(sum(dst["P"][i, src["first"][i]] >= threshold for i in np.where(m)[0]))
    n = int(m.sum())
    return {"transferred": [k, n], "wilson": wilson(k, n)}


def clean_metrics(det, data_path, split_seed=0, val_size=0.15, test_size=0.15, threshold=THRESHOLD):
    """Accuracy, attack F1, attack recall and benign FPR on the fixed clean test split."""
    X, y, _, class_names = load_dataset(data_path)
    _, _, (Xte, yte) = train_val_test_split(X, y, val_size=val_size, test_size=test_size, seed=split_seed)
    flag = det.prob_benign(Xte) < threshold
    mal = yte != class_names.index("benign")
    tp, fp, fn = int((flag & mal).sum()), int((flag & ~mal).sum()), int((~flag & mal).sum())
    return {"n_test": int(len(yte)), "n_attack": int(mal.sum()),
            "accuracy": float((flag == mal).mean()), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "recall": tp / max(int(mal.sum()), 1), "benign_fpr": fp / max(int((~mal).sum()), 1)}
