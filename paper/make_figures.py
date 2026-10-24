"""Render the paper's figures from the committed results. Publication style: Okabe-Ito
colorblind-safe palette, thin marks, recessive grid, single axis, direct labels. Outputs vector PDF
(for LaTeX) + PNG preview into paper/figures/.

    .venv/bin/python paper/make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `evasion` imports

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evasion.detector import Detector

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette (identity by entity, fixed order)
BLUE = "#0072B2"      # classical MLP
VERM = "#D55E00"      # quantum VQC
GREY = "#555555"
THRESH = "#999999"

# Class identity for the data-space figures. Same CVD-safe palette; marker shape and fill style
# also distinguish the classes, so identity never relies on color alone.
BENIGN = "#0072B2"
C2 = "#D55E00"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def fig_evasion_curve():
    """Fig 1: P(benign) vs timing jitter for both detectors, scored on the same swept flows."""
    sweep = np.load("results/evasion/sweep.npz")
    jit = sweep["jitters"]
    feats, got = sweep["features"], sweep["got"]
    mlp = Detector.from_result("results/classical_native")
    vqc = Detector.from_result("results/quantum")

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for det, color, label in [(mlp, BLUE, "Classical MLP"), (vqc, VERM, "Quantum VQC")]:
        means, stds = [], []
        for ji in range(len(jit)):
            ps = [det.prob_benign(feats[si, ji])[0] for si in range(feats.shape[0]) if got[si, ji]]
            means.append(np.mean(ps)); stds.append(np.std(ps))
        means, stds = np.array(means), np.array(stds)
        ax.fill_between(jit, means - stds, means + stds, color=color, alpha=0.13, linewidth=0)
        ax.plot(jit, means, color=color, lw=1.8, marker="o", ms=4, label=label)

    ax.axhline(0.5, color=THRESH, lw=1.0, ls="--")
    ax.text(jit[-1], 0.52, "decision threshold", ha="right", va="bottom", color=THRESH, fontsize=7)
    ax.set_xlabel("Timing jitter (fraction of interval)")
    ax.set_ylabel("P(benign)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(jit[0], jit[-1])
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig1_evasion_curve")


def fig_arms_race():
    """Fig 2: evasion rate across the defense arms race (vs the classical MLP)."""
    labels = ["Timing jitter\nvs. original", "Timing jitter\nvs. hardened",
              "Size padding\nvs. hardened"]
    rates = [100, 0, 100]
    colors = [BLUE, GREY, BLUE]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    bars = ax.bar(labels, rates, color=colors, width=0.62)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 2, f"{r}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Evasion rate")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_arms_race")


def fig_feature_overlap():
    """Fig 3: why clean accuracy is realistic (not saturated). Marginal distributions of the two
    evasion-axis features by class; the rigid-benign confuser overlaps the C2 cluster at low values,
    so a detector that catches rigid beacons necessarily mislabels some benign traffic."""
    data = np.load("data/dataset.npz", allow_pickle=False)
    X, y = data["X"], data["y"]
    names = data["feature_names"].tolist()

    panels = [("cv_iat", "cv_iat  (timing regularity)"),
              ("std_packet_size", "std_packet_size  (bytes)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, (feat, xlabel) in zip(axes, panels):
        j = names.index(feat)
        xb, xc = X[y == 0, j], X[y == 1, j]
        hi = float(np.percentile(np.concatenate([xb, xc]), 98))
        bins = np.linspace(0, hi, 26)
        ax.hist(xb, bins=bins, density=True, color=BENIGN, alpha=0.5,
                edgecolor="white", linewidth=0.3, label="benign")
        ax.hist(xc, bins=bins, density=True, color=C2, alpha=0.5, hatch="///",
                edgecolor=C2, linewidth=0.0, label="C2 beacon")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_xlim(0, hi)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig3_feature_overlap")


def fig_decision_space():
    """Fig 4: the evasion mechanism as a 2D projection onto the two evasion axes. C2 beacons sit at
    low cv_iat / low std_packet_size; raising jitter (and size padding) walks them along these axes
    into the benign-populated region."""
    data = np.load("data/dataset.npz", allow_pickle=False)
    X, y = data["X"], data["y"]
    names = data["feature_names"].tolist()
    jx, jy = names.index("cv_iat"), names.index("std_packet_size")
    xb, yb = X[y == 0, jx], X[y == 0, jy]
    xc, yc = X[y == 1, jx], X[y == 1, jy]

    xmax = float(np.percentile(np.concatenate([xb, xc]), 98))
    ymax = float(np.percentile(np.concatenate([yb, yc]), 98))

    fig, ax = plt.subplots(figsize=(3.9, 3.1))
    ax.scatter(xb, yb, s=16, c=BENIGN, marker="o", alpha=0.55, linewidths=0, label="benign")
    ax.scatter(xc, yc, s=28, c=C2, marker="^", alpha=0.85, linewidths=0, label="C2 beacon")

    # Evasion direction: raising jitter increases cv_iat, moving the beacon toward the benign cloud.
    x0, yarrow = float(np.median(xc)), float(np.median(yc))
    x1 = float(np.percentile(xb, 70))
    ax.annotate("", xy=(x1, yarrow), xytext=(x0, yarrow),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6))
    ax.text((x0 + x1) / 2, yarrow + 0.05 * ymax, "+ timing jitter",
            ha="center", va="bottom", color=GREY, fontsize=7.5)

    ax.set_xlim(-0.03, xmax)
    ax.set_ylim(-0.04 * ymax, ymax)
    ax.set_xlabel("cv_iat  (timing regularity)")
    ax.set_ylabel("std_packet_size  (bytes)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_decision_space")


if __name__ == "__main__":
    fig_evasion_curve()
    fig_arms_race()
    fig_feature_overlap()
    fig_decision_space()
