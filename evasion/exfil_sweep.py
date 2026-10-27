"""Capture-once sweep for the EXFIL paired classical-vs-quantum comparison (PROTOTYPE). Runs the FULL
throttle grid for every exfil INSTANCE and saves each flow's features + functional verdict to disk —
WITHOUT any detector. Scoring happens offline (evasion/exfil_transfer.py), so all detectors are
compared on the identical set of captured flows (a paired comparison) and transferability can be
measured. Mirrors evasion/sweep.py (the C2 version). Run as root:

    sudo .venv/bin/python -m evasion.exfil_sweep --volumes 800000,1100000,1300000,1500000 \
         --seeds-per-config 5 --out results/evasion/exfil_sweep.npz

An INSTANCE is a (base_config, seed) pair: the base_config fixes the exfil's operating point (here the
payload volume, swept across --volumes within the trained band) and the seed is the RNG draw within it.
This replaces the earlier single-volume, seed-only design so the result can be reported across operating
points instead of one repeated configuration.

Expensive: instances x grid live captures (no early stop). 20 instances x 15 throttles x ~70s is
several hours — wrap in systemd-inhibit and leave it. Reduce --seeds-per-config or the grid for a
faster pass.
"""
import argparse

import numpy as np

from evasion.exfil_query import run_query, effective_rate
from evasion.configs import build_configs, expand_instances
from features.dataset import FEATURE_NAMES

# finer grid near the completion/decision region: the old 1.0->2.0 gap put every exfil's min throttle
# exactly on 2.0 (true value in (1,2]), so steps are 0.25 through 3.0, coarser out to 30.
DEFAULT_THROTTLES = "0.0,0.5,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,8.0,12.0,20.0,30.0"


def _ckpt_path(out):
    base = out[:-4] if out.endswith(".npz") else out
    return base + ".partial.npz"


def _atomic_savez(path, **arrays):
    """Write to a temp file then rename, so a power cut mid-write never corrupts the checkpoint."""
    import os
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _resume(out, grid, labels, arrays):
    """If a checkpoint for this exact sweep exists, load its arrays into `arrays` (in place) and return
    the per-instance done mask; otherwise return None. A checkpoint from a different grid or instance
    set is refused rather than silently mixed in."""
    import os
    path = _ckpt_path(out)
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    same = (np.array_equal(d["grid"], np.array(grid))
            and np.array_equal(d["inst_seed"], np.array(labels["inst_seed"]))
            and np.array_equal(d["inst_config_idx"], np.array(labels["inst_config_idx"])))
    if not same:
        raise SystemExit(f"checkpoint {path} is from a different sweep (grid/instances differ); "
                         "move it aside or use another --out")
    for k in arrays:
        arrays[k][...] = d[k]
    return d["inst_done"].astype(bool)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volumes", default="800000,1100000,1300000,1500000",
                   help="payload volumes (bytes) to diversify base configs over (kept in the trained band)")
    p.add_argument("--seeds-per-config", type=int, default=5, help="RNG seeds per base config")
    p.add_argument("--rate", type=float, default=600_000.0, help="fixed naive base rate (pre-throttle)")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--jitter", type=float, default=0.1)
    p.add_argument("--deadline", type=float, default=60.0)
    p.add_argument("--throttles", default=DEFAULT_THROTTLES)
    p.add_argument("--window", type=float, default=65.0)
    p.add_argument("--out", default="results/evasion/exfil_sweep.npz")
    p.add_argument("--work-dir", default="/tmp/exfil_evasion_work")
    p.add_argument("--topology", default="full", choices=["full", "minimal"],
                   help="full = training scenario network with background traffic (in-distribution); "
                        "minimal = old 2-host topology")
    args = p.parse_args()

    grid = [float(x) for x in args.throttles.split(",")]
    volumes = [int(x) for x in args.volumes.split(",")]
    base = {"rate": args.rate, "chunk_size": args.chunk_size, "jitter": args.jitter,
            "deadline": args.deadline}
    configs = build_configs(base, volume=volumes)
    instances, labels = expand_instances(configs, args.seeds_per_config)

    n_i, n_t, n_f = len(instances), len(grid), len(FEATURE_NAMES)
    features = np.full((n_i, n_t, n_f), np.nan)
    functional = np.zeros((n_i, n_t), dtype=bool)
    got = np.zeros((n_i, n_t), dtype=bool)   # whether a flow was actually extracted
    # completion time (s from window start; nan = never completed) and receiver-side confirmation
    # (1 yes, 0 no, -1 no sink log). Stored so evasion/frontier.py can re-judge functionality for ANY
    # deadline <= --deadline offline: a transfer that completes by D has the same flow either way.
    completion_time = np.full((n_i, n_t), np.nan)
    confirmed = np.full((n_i, n_t), -1, dtype=np.int8)

    def one(cfg, throttle, seed):
        theta = {**cfg, "throttle": throttle}
        return run_query(theta, window=args.window, seed=seed, work_dir=args.work_dir,
                         topology=args.topology)

    # checkpoint/resume: progress is saved after every instance to <out>.partial.npz; rerunning the
    # same command after a crash or power cut skips finished instances instead of starting over.
    state = {"features": features, "functional": functional, "got": got, "completion_time": completion_time, "delivery_confirmed": confirmed}
    inst_done = _resume(args.out, grid, labels, state)
    if inst_done is None:
        inst_done = np.zeros(len(instances), dtype=bool)
    else:
        print(f"resuming from checkpoint: {int(inst_done.sum())}/{len(instances)} instances already done\n")

    def checkpoint():
        _atomic_savez(_ckpt_path(args.out), grid=np.array(grid), inst_done=inst_done,
                      inst_seed=np.array(labels["inst_seed"]),
                      inst_config_idx=np.array(labels["inst_config_idx"]), **state)

    # preflight so a dead OVS fails in ~1 min, not after a wasted hour
    print("preflight capture (naive, throttle=0)...")
    pf = one(instances[0][1], grid[0], instances[0][2])
    if pf.features is None:
        raise SystemExit("preflight produced no flow — is OVS up? "
                         "sudo systemctl start ovsdb-server ovs-vswitchd")
    print(f"preflight ok (functional={pf.functionality.functional})\n")

    total = n_i * n_t
    done = int(inst_done.sum()) * n_t
    for ii, (ci, cfg, seed) in enumerate(instances):
        if inst_done[ii]:
            continue
        for ti, t in enumerate(grid):
            try:
                r = one(cfg, t, seed)
            except Exception as e:  # noqa: BLE001 - keep the campaign alive through a flaky capture
                print(f"  inst {ii} (cfg {ci} vol={cfg['volume']} seed={seed}) throttle {t}: FAILED ({e})")
                continue
            if r.features is not None:
                features[ii, ti] = r.features
                functional[ii, ti] = r.functionality.functional
                got[ii, ti] = True
            f = r.functionality
            if f.completion_time is not None:
                completion_time[ii, ti] = f.completion_time
            if f.delivery_confirmed is not None:
                confirmed[ii, ti] = int(f.delivery_confirmed)
            done += 1
            er = effective_rate({"rate": cfg["rate"], "throttle": t})
            print(f"  [{done:3d}/{total}] inst {ii:2d} (cfg {ci} vol={cfg['volume']} seed={seed:2d}) "
                  f"throttle {t:<4} (~{er:.0f}B/s) {'flow' if got[ii, ti] else 'NOflow'} "
                  f"functional={functional[ii, ti]}")
        inst_done[ii] = True
        checkpoint()

    np.savez(args.out, features=features, functional=functional, got=got,
             completion_time=completion_time, delivery_confirmed=confirmed,
             volumes=np.array([cfg["volume"] for _, cfg, _ in instances]),
             throttles=np.array(grid), feature_names=np.array(FEATURE_NAMES),
             config_table=np.array(labels["config_table"]),
             inst_config_idx=np.array(labels["inst_config_idx"]),
             inst_seed=np.array(labels["inst_seed"]),
             topology=args.topology, schema_version=3, rate=args.rate, deadline=args.deadline, window=args.window)
    import os
    os.remove(_ckpt_path(args.out))   # final file written; checkpoint no longer needed
    print(f"\nsaved sweep ({int(got.sum())}/{total} flows captured, {n_i} instances over "
          f"{len(configs)} configs) -> {args.out}")


if __name__ == "__main__":
    main()
