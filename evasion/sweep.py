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

Axis escalation (size padding against the jitter-hardened detectors) is the same capture-once sweep on
the other axis, restricted to the held-out instances so nothing overlaps adversarial training, then
scored offline against EVERY hardened detector (evasion/score.py --heldout):

    sudo .venv/bin/python -m evasion.sweep --intervals 2,4,8 --seeds-per-config 10 \
         --axis size_jitter --fixed-jitter 0.5 --only-heldout \
         --values 0,0.05,0.1,0.15,0.2,0.3,0.4,0.6 --out results/evasion/sweep_sizeaxis.npz

Box mode samples timing jitter and size jitter together instead of sweeping one line. Each instance gets
--box N points drawn by Latin hypercube sampling over [--box-jitter] x [--box-size] (one point per
stratum on each axis, RNG seeded by --box-seed and the instance seed), so the captures cover the whole
region the attacker controls rather than two lines through it. Train-instance points feed range-wide
hardening (evasion/joint_harden.py --extra-sweep); held-out points are its test set:

    sudo .venv/bin/python -m evasion.sweep --intervals 2,4,8 --seeds-per-config 10 \
         --box 8 --out results/evasion/sweep_box.npz

Points outside that box test whether range-wide hardening extends past what it was trained on. Two
held-out sweeps cover the L-shaped region around it: timing jitter past 0.8 (any size jitter up to 1.0),
and size jitter past 0.6 with timing jitter inside the box. Size jitter 1.0 still sends >= 1 byte:

    sudo .venv/bin/python -m evasion.sweep --intervals 2,4,8 --seeds-per-config 10 --only-heldout \
         --box 8 --box-jitter 0.8,2.0 --box-size 0,1.0 --box-seed 1 --out results/evasion/sweep_outside_jitter.npz
    sudo .venv/bin/python -m evasion.sweep --intervals 2,4,8 --seeds-per-config 10 --only-heldout \
         --box 8 --box-jitter 0,0.8 --box-size 0.6,1.0 --box-seed 2 --out results/evasion/sweep_outside_size.npz
"""
import argparse

import numpy as np

from evasion.query import run_query
from evasion.configs import build_configs, expand_instances
from features.dataset import FEATURE_NAMES

# finer grid near the decision boundary: the old 0->0.2 gap made "median min jitter" an upper bound at
# grid resolution, so steps are 0.05 through 0.5 where the flip happens, coarser out to 2.0.
DEFAULT_JITTERS = "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.8,1.2,2.0"


def _ckpt_path(out):
    base = out[:-4] if out.endswith(".npz") else out
    return base + ".partial.npz"


def _atomic_savez(path, **arrays):
    """Write to a temp file then rename, so a power cut mid-write never corrupts the checkpoint."""
    import os
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _resume(out, grid, labels, arrays, theta_grid=None):
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
            and np.array_equal(d["inst_config_idx"], np.array(labels["inst_config_idx"]))
            and (theta_grid is None or "theta_grid" not in d or np.allclose(d["theta_grid"], theta_grid)))
    if not same:
        raise SystemExit(f"checkpoint {path} is from a different sweep (grid/instances differ); "
                         "move it aside or use another --out")
    for k in arrays:
        arrays[k][...] = d[k]
    return d["inst_done"].astype(bool)


def box_points(n, inst_seed, jitter_range, size_range, box_seed=0):
    """Latin-hypercube sample of n (jitter, size_jitter) points for one instance: each axis is cut into
    n equal strata, every stratum gets exactly one point, and the pairing across axes is a random
    permutation. Deterministic in (box_seed, inst_seed)."""
    rng = np.random.default_rng([box_seed, int(inst_seed)])
    out = np.empty((n, 2))
    for k, (lo, hi) in enumerate((jitter_range, size_range)):
        u = (rng.permutation(n) + rng.random(n)) / n
        out[:, k] = lo + u * (hi - lo)
    return np.round(out, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intervals", default="2,4,6,8",
                   help="check-in intervals (s) to diversify base configs over")
    p.add_argument("--seeds-per-config", type=int, default=5, help="RNG seeds per base config")
    p.add_argument("--payload-size", type=int, default=64, help="fixed check-in payload (bytes)")
    p.add_argument("--size-jitter", type=float, default=0.0,
                   help="fixed size jitter when sweeping jitter")
    p.add_argument("--jitters", default=DEFAULT_JITTERS, help="grid when --axis jitter")
    p.add_argument("--axis", default="jitter", choices=["jitter", "size_jitter"],
                   help="realizable knob to sweep (size_jitter = axis escalation)")
    p.add_argument("--values", default=None, help="grid for --axis (defaults to --jitters)")
    p.add_argument("--fixed-jitter", type=float, default=0.5,
                   help="held timing jitter when sweeping size_jitter (an already-evading value)")
    p.add_argument("--box", type=int, default=0,
                   help="box mode: N Latin-hypercube (jitter, size_jitter) points per instance instead of a line")
    p.add_argument("--box-jitter", default="0,0.8", help="timing-jitter range for --box (lo,hi)")
    p.add_argument("--box-size", default="0,0.6", help="size-jitter range for --box (lo,hi)")
    p.add_argument("--box-seed", type=int, default=0, help="sampling seed for --box")
    p.add_argument("--only-heldout", action="store_true",
                   help="capture only the instances evasion.harden holds out (same --train-frac)")
    p.add_argument("--train-frac", type=float, default=0.67, help="must match evasion.harden")
    p.add_argument("--window", type=float, default=60.0)
    p.add_argument("--max-interval", type=float, default=30.0)
    p.add_argument("--topology", default="full", choices=["full", "minimal"],
                   help="full = training scenario network with background traffic (in-distribution); "
                        "minimal = old 2-host topology")
    p.add_argument("--out", default="results/evasion/sweep.npz")
    p.add_argument("--work-dir", default="/tmp/evasion_work")
    args = p.parse_args()

    axis = "box" if args.box else args.axis
    grid = (list(range(args.box)) if args.box
            else [float(x) for x in (args.values or args.jitters).split(",")])
    intervals = [float(x) for x in args.intervals.split(",")]
    if axis == "box":
        base = {"payload_size": args.payload_size}
    elif axis == "jitter":
        base = {"size_jitter": args.size_jitter, "payload_size": args.payload_size}
    else:
        base = {"jitter": args.fixed_jitter, "payload_size": args.payload_size}
    configs = build_configs(base, interval=intervals)
    instances, labels = expand_instances(configs, args.seeds_per_config)
    if args.only_heldout:
        from evasion.harden import split_instances
        _, keep = split_instances({"features": np.zeros((len(instances), 1, 1)),
                                   "inst_config_idx": np.array(labels["inst_config_idx"]),
                                   "inst_seed": np.array(labels["inst_seed"])}, args.train_frac)
        instances = [instances[i] for i in keep]
        labels = {"config_table": labels["config_table"],
                  "inst_config_idx": [labels["inst_config_idx"][i] for i in keep],
                  "inst_seed": [labels["inst_seed"][i] for i in keep]}
        print(f"--only-heldout: capturing {len(instances)} held-out instances")

    n_i, n_j, n_f = len(instances), len(grid), len(FEATURE_NAMES)
    # per-flow (jitter, size_jitter) actually captured; for line sweeps it is the line itself
    theta_grid = np.zeros((n_i, n_j, 2))
    for ii, (_, cfg, seed) in enumerate(instances):
        if axis == "box":
            rng_j = tuple(float(x) for x in args.box_jitter.split(","))
            rng_s = tuple(float(x) for x in args.box_size.split(","))
            theta_grid[ii] = box_points(n_j, seed, rng_j, rng_s, args.box_seed)
        else:
            for ji, v in enumerate(grid):
                t = {**cfg, axis: v}
                theta_grid[ii, ji] = (t.get("jitter", 0.0), t.get("size_jitter", 0.0))
    features = np.full((n_i, n_j, n_f), np.nan)
    functional = np.zeros((n_i, n_j), dtype=bool)
    got = np.zeros((n_i, n_j), dtype=bool)   # whether a flow was actually extracted

    def one(ii, cfg, ji, seed):
        theta = {**cfg, "jitter": float(theta_grid[ii, ji, 0]), "size_jitter": float(theta_grid[ii, ji, 1])}
        return run_query(theta, window=args.window, max_interval=args.max_interval,
                         seed=seed, work_dir=args.work_dir, topology=args.topology)

    # checkpoint/resume: progress is saved after every instance to <out>.partial.npz; rerunning the
    # same command after a crash or power cut skips finished instances instead of starting over.
    state = {"features": features, "functional": functional, "got": got}
    inst_done = _resume(args.out, grid, labels, state, theta_grid)
    if inst_done is None:
        inst_done = np.zeros(len(instances), dtype=bool)
    else:
        print(f"resuming from checkpoint: {int(inst_done.sum())}/{len(instances)} instances already done\n")

    def checkpoint():
        _atomic_savez(_ckpt_path(args.out), grid=np.array(grid), inst_done=inst_done,
                      inst_seed=np.array(labels["inst_seed"]),
                      inst_config_idx=np.array(labels["inst_config_idx"]), theta_grid=theta_grid, **state)

    # preflight so a dead OVS fails in ~1 min, not after a wasted hour
    print("preflight capture...")
    pf = one(0, instances[0][1], 0, instances[0][2])
    if pf.features is None:
        raise SystemExit("preflight produced no flow — is OVS up? "
                         "sudo systemctl start ovsdb-server ovs-vswitchd")
    print(f"preflight ok (functional={pf.functionality.functional})\n")

    total = n_i * n_j
    done = int(inst_done.sum()) * n_j
    for ii, (ci, cfg, seed) in enumerate(instances):
        if inst_done[ii]:
            continue
        for ji, j in enumerate(grid):
            try:
                r = one(ii, cfg, ji, seed)
            except Exception as e:  # noqa: BLE001 - keep the campaign alive through a flaky capture
                print(f"  inst {ii} (cfg {ci} interval={cfg['interval']} seed={seed}) point {ji} "
                      f"{theta_grid[ii, ji].tolist()}: FAILED ({e})")
                continue
            if r.features is not None:
                features[ii, ji] = r.features
                functional[ii, ji] = r.functionality.functional
                got[ii, ji] = True
            done += 1
            print(f"  [{done:3d}/{total}] inst {ii:2d} (cfg {ci} int={cfg['interval']:<4} seed={seed:2d}) "
                  f"jitter {theta_grid[ii, ji, 0]:<6g} size {theta_grid[ii, ji, 1]:<6g} "
                  f"{'flow' if got[ii, ji] else 'NOflow'} functional={functional[ii, ji]}")
        inst_done[ii] = True
        checkpoint()

    grid_key = {"jitters": np.array(grid)} if axis == "jitter" else {}
    np.savez(args.out, features=features, functional=functional, got=got, **grid_key,
             values=np.array(grid), axis=axis, fixed_jitter=args.fixed_jitter,
             heldout_only=args.only_heldout, feature_names=np.array(FEATURE_NAMES), theta_grid=theta_grid,
             box_jitter=args.box_jitter if args.box else "", box_size=args.box_size if args.box else "",
             box_seed=args.box_seed,
             config_table=np.array(labels["config_table"]),
             inst_config_idx=np.array(labels["inst_config_idx"]),
             inst_seed=np.array(labels["inst_seed"]),
             topology=args.topology,
             schema_version=3, window=args.window, max_interval=args.max_interval)
    import os
    os.remove(_ckpt_path(args.out))   # final file written; checkpoint no longer needed
    print(f"\nsaved sweep ({int(got.sum())}/{total} flows captured, {n_i} instances over "
          f"{len(configs)} configs) -> {args.out}")


if __name__ == "__main__":
    main()
