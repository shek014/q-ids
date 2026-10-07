"""Capture-once sweep for the EXFIL paired classical-vs-quantum comparison (PROTOTYPE). Runs the FULL
throttle grid for every exfil seed and saves each flow's features + functional verdict to disk —
WITHOUT any detector. Scoring happens offline (evasion/exfil_transfer.py), so all detectors are
compared on the identical set of captured flows (a paired comparison) and transferability can be
measured. Mirrors evasion/sweep.py (the C2 version). Run as root:

    sudo .venv/bin/python -m evasion.exfil_sweep --seeds 15 --out results/evasion/exfil_sweep.npz

Expensive: seeds x grid live captures (no early stop). ~15 x 10 x ~70s is a few hours — wrap in
systemd-inhibit and leave it. Reduce --seeds or the grid for a faster (weaker) pass.
"""
import argparse

import numpy as np

from evasion.exfil_query import run_query, effective_rate
from features.dataset import FEATURE_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--volume", type=int, default=1_200_000)
    p.add_argument("--rate", type=float, default=600_000.0, help="naive base rate (pre-throttle)")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--jitter", type=float, default=0.1)
    p.add_argument("--deadline", type=float, default=60.0)
    p.add_argument("--throttles", default="0.0,0.5,1.0,2.0,3.0,5.0,8.0,12.0,20.0,30.0")
    p.add_argument("--window", type=float, default=65.0)
    p.add_argument("--out", default="results/evasion/exfil_sweep.npz")
    p.add_argument("--work-dir", default="/tmp/exfil_evasion_work")
    args = p.parse_args()

    grid = [float(x) for x in args.throttles.split(",")]
    n_s, n_t, n_f = args.seeds, len(grid), len(FEATURE_NAMES)
    features = np.full((n_s, n_t, n_f), np.nan)
    functional = np.zeros((n_s, n_t), dtype=bool)
    got = np.zeros((n_s, n_t), dtype=bool)   # whether a flow was actually extracted

    def one(seed, throttle):
        theta = {"volume": args.volume, "rate": args.rate, "throttle": throttle,
                 "chunk_size": args.chunk_size, "jitter": args.jitter, "deadline": args.deadline}
        return run_query(theta, window=args.window, seed=seed, work_dir=args.work_dir)

    # preflight so a dead OVS fails in ~1 min, not after a wasted hour
    print("preflight capture (naive, throttle=0)...")
    pf = one(0, grid[0])
    if pf.features is None:
        raise SystemExit("preflight produced no flow — is OVS up? "
                         "sudo systemctl start ovsdb-server ovs-vswitchd")
    print(f"preflight ok (functional={pf.functionality.functional})\n")

    total = n_s * n_t
    done = 0
    for si in range(n_s):
        for ti, t in enumerate(grid):
            try:
                r = one(si, t)
            except Exception as e:  # noqa: BLE001 - keep the campaign alive through a flaky capture
                print(f"  seed {si} throttle {t}: FAILED ({e})")
                continue
            if r.features is not None:
                features[si, ti] = r.features
                functional[si, ti] = r.functionality.functional
                got[si, ti] = True
            done += 1
            er = effective_rate({"rate": args.rate, "throttle": t})
            print(f"  [{done:3d}/{total}] seed {si:2d} throttle {t:<4} (~{er:.0f}B/s) "
                  f"{'flow' if got[si, ti] else 'NOflow'} functional={functional[si, ti]}")

    np.savez(args.out, features=features, functional=functional, got=got,
             throttles=np.array(grid), seeds=np.arange(n_s), feature_names=np.array(FEATURE_NAMES),
             volume=args.volume, rate=args.rate, deadline=args.deadline, window=args.window)
    print(f"\nsaved sweep ({int(got.sum())}/{total} flows captured) -> {args.out}")


if __name__ == "__main__":
    main()
