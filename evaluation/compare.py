import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import summarize


def load_run(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    history = dict(np.load(run_dir / "history.npz"))
    preds = np.load(run_dir / "test_predictions.npz")
    return meta, history, preds["y_true"], preds["y_proba"]


def print_table(c_meta, c_summary, q_meta, q_summary):
    rows = [
        ("accuracy", c_summary["accuracy"], q_summary["accuracy"], ".4f"),
        ("macro f1", c_summary["macro_f1"], q_summary["macro_f1"], ".4f"),
        ("macro FPR", c_summary["macro_false_positive_rate"], q_summary["macro_false_positive_rate"], ".4f"),
        ("params", c_meta["param_count"], q_meta["param_count"], ""),
        ("train time (s)", c_meta["train_time_seconds"], q_meta["train_time_seconds"], ".2f"),
    ]
    print(f"{'metric':<18}{'classical':>14}{'quantum':>14}")
    for name, c_val, q_val, fmt in rows:
        c_str = format(c_val, fmt) if fmt else str(c_val)
        q_str = format(q_val, fmt) if fmt else str(q_val)
        print(f"{name:<18}{c_str:>14}{q_str:>14}")

    print(f"\n{'class':<10}{'classical f1':>14}{'quantum f1':>14}"
          f"{'classical FPR':>16}{'quantum FPR':>14}")
    for name in c_summary["per_class"]:
        c_pc, q_pc = c_summary["per_class"][name], q_summary["per_class"][name]
        print(f"{name:<10}{c_pc['f1']:>14.4f}{q_pc['f1']:>14.4f}"
              f"{c_pc['false_positive_rate']:>16.4f}{q_pc['false_positive_rate']:>14.4f}")


def plot(c_hist, q_hist, c_summary, q_summary, class_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, ax, title in [("loss", axes[0], "loss"), ("acc", axes[1], "accuracy")]:
        ax.plot(c_hist[f"train_{key}"], label="classical train")
        ax.plot(c_hist[f"val_{key}"], label="classical val")
        ax.plot(q_hist[f"train_{key}"], label="quantum train")
        ax.plot(q_hist[f"val_{key}"], label="quantum val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, summary, title in [(axes[0], c_summary, "classical"), (axes[1], q_summary, "quantum")]:
        cm = np.array(summary["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_title(title)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results", help="dir containing classical/ and quantum/ run outputs")
    parser.add_argument("--out", default="results/comparison")
    args = parser.parse_args()

    runs = Path(args.runs)
    c_meta, c_hist, c_yt, c_proba = load_run(runs / "classical")
    q_meta, q_hist, q_yt, q_proba = load_run(runs / "quantum")

    class_names = c_meta["class_names"]
    c_summary = summarize(c_yt, c_proba, class_names)
    q_summary = summarize(q_yt, q_proba, class_names)

    print_table(c_meta, c_summary, q_meta, q_summary)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({
        "classical": {"summary": c_summary, "param_count": c_meta["param_count"],
                      "train_time_seconds": c_meta["train_time_seconds"]},
        "quantum": {"summary": q_summary, "param_count": q_meta["param_count"],
                    "train_time_seconds": q_meta["train_time_seconds"]},
    }, indent=2))
    plot(c_hist, q_hist, c_summary, q_summary, class_names, out)
    print(f"\nsaved comparison to {out}")


if __name__ == "__main__":
    main()
