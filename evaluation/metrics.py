import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.clip(tp + fp, 1e-12, None)
    recall = tp / np.clip(tp + fn, 1e-12, None)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return precision, recall, f1


def false_positive_rate(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    return fp / np.clip(fp + tn, 1e-12, None)


def roc_curve(y_true_binary, scores):
    order = np.argsort(-scores)
    y_sorted = y_true_binary[order]
    n_pos = max(y_sorted.sum(), 1)
    n_neg = max(len(y_sorted) - y_sorted.sum(), 1)
    tpr = np.concatenate([[0], np.cumsum(y_sorted) / n_pos])
    fpr = np.concatenate([[0], np.cumsum(1 - y_sorted) / n_neg])
    return fpr, tpr


def pr_curve(y_true_binary, scores):
    order = np.argsort(-scores)
    y_sorted = y_true_binary[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.clip(tp + fp, 1e-12, None)
    recall = tp / max(y_sorted.sum(), 1)
    return np.concatenate([[0.0], recall]), np.concatenate([[1.0], precision])


def roc_auc_score(y_true_binary, scores):
    fpr, tpr = roc_curve(y_true_binary, scores)
    return float(np.trapz(tpr, fpr))


def pr_auc_score(y_true_binary, scores):
    recall, precision = pr_curve(y_true_binary, scores)
    return float(np.trapz(precision, recall))


def summarize(y_true, y_proba, class_names):
    n_classes = len(class_names)
    y_pred = np.argmax(y_proba, axis=1)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, n_classes)
    fpr = false_positive_rate(y_true, y_pred, n_classes)

    per_class = {}
    for c, name in enumerate(class_names):
        y_binary = (y_true == c).astype(int)
        per_class[name] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "false_positive_rate": float(fpr[c]),
            "roc_auc": roc_auc_score(y_binary, y_proba[:, c]),
            "pr_auc": pr_auc_score(y_binary, y_proba[:, c]),
        }

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1)),
        "macro_false_positive_rate": float(np.mean(fpr)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes).tolist(),
    }
