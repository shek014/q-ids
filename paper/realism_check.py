"""Realism check: is the synthetic benign traffic's timing regularity something real benign traffic
actually shows? Compares cv_iat (timing regularity, the C2 detection signal) for real benign flows from a
public capture against the synthetic benign and C2 flows in data/dataset.npz, and writes
paper/figures/fig6_realism.pdf.

Real flows come from data/real_benign.npz: a public normal-host capture, filtered to the host's outbound
packets and cut into 60 s windows (the scenario window), then run through features.extract.
Only flows with >= 5 packets are compared, since cv_iat is meaningless for a handful of packets.

    .venv/bin/python paper/realism_check.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_PKTS = 5
syn = np.load("data/dataset.npz")
real = np.load("data/real_benign.npz")
fn = list(syn["feature_names"])
cv, pk = fn.index("cv_iat"), fn.index("packet_count")


def cv_of(X):
    X = X[X[:, pk] >= MIN_PKTS]
    return X[:, cv]


sets = {"real benign": cv_of(real["X"]),
        "synthetic benign": cv_of(syn["X"][syn["y"] == 0]),
        "synthetic C2": cv_of(syn["X"][syn["y"] == 1])}
beacon_hi = np.percentile(sets["synthetic C2"], 95)          # upper edge of the beacon band

print(f"flows with >= {MIN_PKTS} packets; beacon band = cv_iat <= {beacon_hi:.3f} (C2 95th pct)\n")
print(f"{'set':18s}{'n':>7s}{'p5':>9s}{'p25':>9s}{'p50':>9s}{'in beacon band':>17s}")
for name, v in sets.items():
    q = np.percentile(v, [5, 25, 50])
    print(f"{name:18s}{len(v):>7d}{q[0]:>9.3f}{q[1]:>9.3f}{q[2]:>9.3f}{(v <= beacon_hi).mean()*100:>16.1f}%")

fig, ax = plt.subplots(figsize=(4.2, 2.8))
bins = np.logspace(-4, 1, 40)
for (name, v), c in zip(sets.items(), ["#4C72B0", "#999999", "#C44E52"]):
    ax.hist(np.clip(v, 1e-4, 10), bins=bins, density=True, histtype="step", lw=1.6, label=name, color=c)
ax.axvline(beacon_hi, ls="--", lw=1, color="k")
ax.set_xscale("log")
ax.set_xlabel("cv_iat (timing irregularity)")
ax.set_ylabel("density")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("paper/figures/fig6_realism.pdf")
print("\nsaved paper/figures/fig6_realism.pdf")
