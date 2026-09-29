"""Capture-once sweep for the paired classical-vs-quantum comparison. Runs the FULL jitter grid for
every beacon seed and saves each flow's features + functional verdict to disk — WITHOUT any detector.
Scoring happens offline (evasion/transfer.py), so both models are compared on the identical set of
captured flows (a paired comparison) and transferability can be measured. Run as root:

    sudo .venv/bin/python -m evasion.sweep --seeds 15 --out results/evasion/sweep.npz

Expensive: seeds x grid live captures (no early stop). ~15 x 12 x ~65s is a few hours — wrap in
systemd-inhibit and leave it. Reduce --seeds for a faster (weaker) pass.
"""
import argparse

import numpy as np

from evasion.query import run_query
from features.dataset import FEATURE_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--interval", type=float, default=4.0)
    p.add_argument("--jitters", default="0.0,0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.6,0.8,1.2,2.0")
    p.add_argument("--window", type=float, default=60.0)
    p.add_argument("--max-interval", type=float, default=30.0)
    p.add_argument("--out", default="results/evasion/sweep.npz")
    p.add_argument("--work-dir", default="/tmp/evasion_work")
    args = p.parse_args()

    grid = [float(x) for x in args.jitters.split(",")]
    n_s, n_j, n_f = args.seeds, len(grid), len(FEATURE_NAMES)
    features = np.full((n_s, n_j, n_f), np.nan)
    functional = np.zeros((n_s, n_j), dtype=bool)
    got = np.zeros((n_s, n_j), dtype=bool)   # whether a flow was actually extracted

    def one(seed, jitter):
        theta = {"interval": args.interval, "jitter": jitter, "size_jitter": 0.0, "payload_size": 64}
        return run_query(theta, window=args.window, max_interval=args.max_interval,
                         seed=seed, work_dir=args.work_dir)

    # preflight so a dead OVS fails in ~1 min, not after a wasted hour
    print("preflight capture...")
    pf = one(0, grid[0])
    if pf.features is None:
        raise SystemExit("preflight produced no flow — is OVS up? "
                         "sudo systemctl start ovsdb-server ovs-vswitchd")
    print(f"preflight ok (functional={pf.functionality.functional})\n")

    total = n_s * n_j
    done = 0
    for si in range(n_s):
        for ji, j in enumerate(grid):
            try:
                r = one(si, j)
            except Exception as e:  # noqa: BLE001 - keep the campaign alive through a flaky capture
                print(f"  seed {si} jitter {j}: FAILED ({e})")
                continue
            if r.features is not None:
                features[si, ji] = r.features
                functional[si, ji] = r.functionality.functional
                got[si, ji] = True
            done += 1
            print(f"  [{done:3d}/{total}] seed {si:2d} jitter {j:<4} "
                  f"{'flow' if got[si, ji] else 'NOflow'} functional={functional[si, ji]}")

    np.savez(args.out, features=features, functional=functional, got=got,
             jitters=np.array(grid), seeds=np.arange(n_s), feature_names=np.array(FEATURE_NAMES),
             interval=args.interval, window=args.window, max_interval=args.max_interval)
    print(f"\nsaved sweep ({int(got.sum())}/{total} flows captured) -> {args.out}")


if __name__ == "__main__":
    main()
