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


if __name__ == "__main__":
    fig_evasion_curve()
    fig_arms_race()
