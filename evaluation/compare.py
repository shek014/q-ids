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
    return {"meta": meta, "history": history, "y_true": preds["y_true"], "y_proba": preds["y_proba"]}


def print_table(runs, class_names):
    labels = list(runs)
    col = max(14, max(len(l) for l in labels) + 2)

    def row(name, values, fmt):
        cells = "".join(f"{(format(v, fmt) if fmt else str(v)):>{col}}" for v in values)
        print(f"{name:<20}{cells}")

    print(f"\n{'metric':<20}" + "".join(f"{l:>{col}}" for l in labels))
    row("accuracy", [runs[l]["summary"]["accuracy"] for l in labels], ".4f")
    row("macro f1", [runs[l]["summary"]["macro_f1"] for l in labels], ".4f")
    row("macro FPR", [runs[l]["summary"]["macro_false_positive_rate"] for l in labels], ".4f")
    row("params", [runs[l]["meta"]["param_count"] for l in labels], "")
    row("n_features", [runs[l]["meta"].get("n_features", "?") for l in labels], "")
    row("train time (s)", [runs[l]["meta"]["train_time_seconds"] for l in labels], ".2f")

    print(f"\n{'per-class f1':<20}" + "".join(f"{l:>{col}}" for l in labels))
    for name in class_names:
        row(name, [runs[l]["summary"]["per_class"][name]["f1"] for l in labels], ".4f")


def plot(runs, class_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(runs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax, title in [("loss", axes[0], "val loss"), ("acc", axes[1], "val accuracy")]:
        for l in labels:
            hist = runs[l]["history"]
            if f"val_{key}" in hist:
                ax.plot(hist[f"val_{key}"], label=l)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(labels), figsize=(4.5 * len(labels), 4))
    if len(labels) == 1:
        axes = [axes]
    for ax, l in zip(axes, labels):
        cm = np.array(runs[l]["summary"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_title(l)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare model runs. The primary head-to-head is MLP-matched vs VQC "
                    "(same PCA features); MLP-native (all raw features) is a reference.")
    parser.add_argument("--classical-matched", default="results/classical_matched")
    parser.add_argument("--quantum", default="results/quantum")
    parser.add_argument("--classical-native", default="results/classical_native")
    parser.add_argument("--out", default="results/comparison")
    args = parser.parse_args()

    # order matters: primary comparison first, reference last
    candidates = [
        ("mlp_matched", args.classical_matched),
        ("vqc", args.quantum),
        ("mlp_native", args.classical_native),
    ]
    runs = {}
    for label, path in candidates:
        if (Path(path) / "meta.json").exists():
            runs[label] = load_run(path)
        else:
            print(f"note: {label} not found at {path}, skipping")

    if len(runs) < 2:
        raise SystemExit("need at least two runs to compare")

    class_names = next(iter(runs.values()))["meta"]["class_names"]
    for r in runs.values():
        r["summary"] = summarize(r["y_true"], r["y_proba"], class_names)

    print_table(runs, class_names)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({
        label: {"summary": r["summary"], "param_count": r["meta"]["param_count"],
                "n_features": r["meta"].get("n_features"),
                "input_mode": r["meta"].get("input_mode"),
                "train_time_seconds": r["meta"]["train_time_seconds"]}
        for label, r in runs.items()
    }, indent=2))
    plot(runs, class_names, out)
    print(f"\nsaved comparison to {out}")


if __name__ == "__main__":
    main()
