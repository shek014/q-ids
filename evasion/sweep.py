"""Capture-once sweep for the paired classical-vs-quantum comparison. Runs the FULL jitter grid for
every beacon INSTANCE and saves each flow's features + functional verdict to disk — WITHOUT any
detector. Scoring happens offline (evasion/transfer.py), so both models are compared on the identical
set of captured flows (a paired comparison) and transferability can be measured. Run as root:

    sudo .venv/bin/python -m evasion.sweep --intervals 2,4,6,8 --seeds-per-config 5 \
         --out results/evasion/sweep.npz

An INSTANCE is a (base_config, seed) pair: the base_config fixes the beacon's operating point (here the
check-in interval, swept across --intervals) and the seed is the RNG draw within it. This replaces the
earlier single-interval, seed-only design so the evasion result can be reported across operating points
instead of one repeated configuration.

Expensive: instances x grid live captures (no early stop). 20 instances x 15 jitters x ~65s is several
hours — wrap in systemd-inhibit and leave it. Reduce --seeds-per-config or the grid for a faster pass.
"""
import argparse

import numpy as np

from evasion.query import run_query
from evasion.configs import build_configs, expand_instances
from features.dataset import FEATURE_NAMES

# finer grid near the decision boundary: the old 0->0.2 gap made "median min jitter" an upper bound at
# grid resolution, so steps are 0.05 through 0.5 where the flip happens, coarser out to 2.0.
DEFAULT_JITTERS = "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.8,1.2,2.0"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intervals", default="2,4,6,8",
                   help="check-in intervals (s) to diversify base configs over")
    p.add_argument("--seeds-per-config", type=int, default=5, help="RNG seeds per base config")
    p.add_argument("--payload-size", type=int, default=64, help="fixed check-in payload (bytes)")
    p.add_argument("--size-jitter", type=float, default=0.0, help="fixed size jitter (jitter is swept)")
    p.add_argument("--jitters", default=DEFAULT_JITTERS)
    p.add_argument("--window", type=float, default=60.0)
    p.add_argument("--max-interval", type=float, default=30.0)
    p.add_argument("--out", default="results/evasion/sweep.npz")
    p.add_argument("--work-dir", default="/tmp/evasion_work")
    args = p.parse_args()

    grid = [float(x) for x in args.jitters.split(",")]
    intervals = [float(x) for x in args.intervals.split(",")]
    base = {"size_jitter": args.size_jitter, "payload_size": args.payload_size}
    configs = build_configs(base, interval=intervals)
    instances, labels = expand_instances(configs, args.seeds_per_config)

    n_i, n_j, n_f = len(instances), len(grid), len(FEATURE_NAMES)
    features = np.full((n_i, n_j, n_f), np.nan)
    functional = np.zeros((n_i, n_j), dtype=bool)
    got = np.zeros((n_i, n_j), dtype=bool)   # whether a flow was actually extracted

    def one(cfg, jitter, seed):
        theta = {**cfg, "jitter": jitter}
        return run_query(theta, window=args.window, max_interval=args.max_interval,
                         seed=seed, work_dir=args.work_dir)

    # preflight so a dead OVS fails in ~1 min, not after a wasted hour
    print("preflight capture...")
    pf = one(instances[0][1], grid[0], instances[0][2])
    if pf.features is None:
        raise SystemExit("preflight produced no flow — is OVS up? "
                         "sudo systemctl start ovsdb-server ovs-vswitchd")
    print(f"preflight ok (functional={pf.functionality.functional})\n")

    total = n_i * n_j
    done = 0
    for ii, (ci, cfg, seed) in enumerate(instances):
        for ji, j in enumerate(grid):
            try:
                r = one(cfg, j, seed)
            except Exception as e:  # noqa: BLE001 - keep the campaign alive through a flaky capture
                print(f"  inst {ii} (cfg {ci} interval={cfg['interval']} seed={seed}) jitter {j}: FAILED ({e})")
                continue
            if r.features is not None:
                features[ii, ji] = r.features
                functional[ii, ji] = r.functionality.functional
                got[ii, ji] = True
            done += 1
            print(f"  [{done:3d}/{total}] inst {ii:2d} (cfg {ci} int={cfg['interval']:<4} seed={seed:2d}) "
                  f"jitter {j:<4} {'flow' if got[ii, ji] else 'NOflow'} functional={functional[ii, ji]}")

    np.savez(args.out, features=features, functional=functional, got=got,
             jitters=np.array(grid), feature_names=np.array(FEATURE_NAMES),
             config_table=np.array(labels["config_table"]),
             inst_config_idx=np.array(labels["inst_config_idx"]),
             inst_seed=np.array(labels["inst_seed"]),
             schema_version=2, window=args.window, max_interval=args.max_interval)
    print(f"\nsaved sweep ({int(got.sum())}/{total} flows captured, {n_i} instances over "
          f"{len(configs)} configs) -> {args.out}")


if __name__ == "__main__":
    main()
