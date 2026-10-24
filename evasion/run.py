"""Evasion experiment driver. For a trained detector, runs the jitter line search over many beacon
INSTANCES and reports the headline metrics: evasion rate, queries-to-evasion, and the minimal evading
jitter (the cost of evasion). An instance is a (base_config, seed) pair — the base config fixes the
check-in interval (swept across --intervals) and the seed is the RNG draw within it — so the result is
reported pooled AND broken down per operating point, not for one repeated configuration. Each query is
a live capture, so run as root:

    sudo .venv/bin/python -m evasion.run --detector results/classical_native \
         --intervals 2,4,6,8 --seeds-per-config 5

Results (per-seed trajectories + pooled/per-config summary) are written to results/evasion/<detector>.json.
"""
import argparse
import json
import statistics
from pathlib import Path

from evasion.detector import Detector
from evasion.query import run_query, QueryResult
from evasion.oracle import Functionality
from evasion.search import line_search
from evasion.configs import build_configs, expand_instances

# a broken capture shouldn't abort a multi-hour run; treat it as a non-evading, non-functional query
_FAILED = QueryResult(theta={}, features=None,
                      functionality=Functionality(False, 0, float("inf"), 0.0))


def _errored(r):
    """An instance errored (infrastructure, not the detector) if no query produced a flow to score."""
    return bool(r.trajectory) and all(p is None for _, p, _ in r.trajectory)


def _pool(results):
    """Headline metrics over a set of instances (only those with real captures count)."""
    scored = [r for r in results if not _errored(r)]
    evaded = [r for r in scored if r.evaded]
    n = len(scored)
    return {
        "n": len(results),
        "n_errored": len(results) - len(scored),
        "evasion_rate": len(evaded) / n if n else 0.0,
        "median_queries_to_evasion": statistics.median([r.queries for r in evaded]) if evaded else None,
        "median_min_jitter": statistics.median([r.min_jitter for r in evaded]) if evaded else None,
    }


def summarize(results, group_key="interval"):
    out = _pool(results)
    out["n_beacons"] = len(results)
    groups = {}
    for r in results:
        groups.setdefault(r.base_theta.get(group_key), []).append(r)
    out["per_config"] = {f"{group_key}={k}": _pool(v) for k, v in sorted(groups.items())}
    out["per_seed"] = [{"seed": r.seed, group_key: r.base_theta.get(group_key),
                        "evaded": r.evaded, "errored": _errored(r),
                        "min_jitter": r.min_jitter, "queries": r.queries, "trajectory": r.trajectory}
                       for r in results]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector", default="results/classical_native", help="a trained result dir")
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--intervals", default="2,4,6,8", help="check-in intervals (s) to diversify over")
    p.add_argument("--seeds-per-config", type=int, default=5, help="RNG seeds per base config")
    p.add_argument("--axis", default="jitter", choices=["jitter", "size_jitter"],
                   help="which realizable knob to line-search (size_jitter = step-4 escalation)")
    p.add_argument("--values", default=None, help="ascending grid for --axis (defaults to --jitters)")
    p.add_argument("--jitters", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.8,1.2,2.0",
                   help="grid used when --values is omitted (finer near the boundary)")
    p.add_argument("--fixed-jitter", type=float, default=0.1,
                   help="held jitter when searching size_jitter (set to an evading value, e.g. 0.5)")
    p.add_argument("--fixed-size-jitter", type=float, default=0.0,
                   help="held size_jitter when searching jitter")
    p.add_argument("--payload-size", type=int, default=64)
    p.add_argument("--window", type=float, default=60.0)
    p.add_argument("--max-interval", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=0.5, help="P(benign) to count as evaded")
    p.add_argument("--full-grid", action="store_true", help="evaluate the whole grid (curve), no early stop")
    p.add_argument("--work-dir", default="/tmp/evasion_work")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    detector = Detector.from_result(args.detector, args.data)
    values = [float(x) for x in (args.values or args.jitters).split(",")]
    intervals = [float(x) for x in args.intervals.split(",")]
    base = {"jitter": args.fixed_jitter, "size_jitter": args.fixed_size_jitter,
            "payload_size": args.payload_size}
    configs = build_configs(base, interval=intervals)
    instances, _ = expand_instances(configs, args.seeds_per_config)

    def query_fn(theta, seed):
        try:
            return run_query(theta, window=args.window, max_interval=args.max_interval,
                             seed=seed, work_dir=args.work_dir)
        except Exception as e:  # noqa: BLE001 - one flaky capture must not kill the batch
            print(f"  query failed (seed={seed}, {args.axis}={theta[args.axis]}): {e}")
            return _FAILED

    print(f"detector '{detector.name}' ({detector.meta['model']}, {detector.meta['input_mode']})  "
          f"clean test acc {detector.meta['test_accuracy']:.3f}")

    # Preflight: one real capture up front. Catches infrastructure failure (OVS down, etc.) in ~1
    # minute instead of letting the whole run report a bogus 0.00 evasion rate an hour later.
    print("preflight capture...")
    pf_cfg, pf_seed = instances[0][1], instances[0][2]
    try:
        pf = run_query({**pf_cfg, args.axis: values[0]}, window=args.window,
                       max_interval=args.max_interval, seed=pf_seed, work_dir=args.work_dir)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"preflight capture FAILED: {e}\n"
                         "Is OVS up? try: sudo systemctl start ovsdb-server ovs-vswitchd")
    if pf.features is None:
        raise SystemExit("preflight produced no beacon flow — check OVS / capture setup before the run")
    print(f"preflight ok: functional={pf.functionality.functional}, "
          f"P(benign)={detector.prob_benign(pf.features)[0]:.3f}\n")
    print(f"line search on {args.axis}: intervals={intervals}, seeds/config={args.seeds_per_config}, "
          f"fixed_jitter={args.fixed_jitter}, values={values}, {len(instances)} instances\n")

    results = []
    for ii, (ci, cfg, seed) in enumerate(instances):
        r = line_search(detector.prob_benign, query_fn, cfg, values, seed=seed,
                        threshold=args.threshold, early_stop=not args.full_grid, axis=args.axis)
        if _errored(r):
            status = "ERRORED (no capture)"
        elif r.evaded:
            status = f"EVADED  min_{args.axis}={r.min_jitter}"
        else:
            status = "held   (detector held)"
        print(f"  inst {ii:2d} (int={cfg['interval']:<4} seed={seed:2d}): {status}  ({r.queries} queries)")
        results.append(r)

    summary = summarize(results, group_key="interval")
    out_path = Path(args.out) if args.out else Path("results/evasion") / f"{detector.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"detector": detector.name, "meta": detector.meta,
                                    "config": vars(args), "summary": summary}, indent=2))

    print(f"\nevasion rate: {summary['evasion_rate']:.2f}  "
          f"median queries-to-evasion: {summary['median_queries_to_evasion']}  "
          f"median min jitter: {summary['median_min_jitter']}")
    for name, s in summary["per_config"].items():
        print(f"  {name}: evasion {s['evasion_rate']:.2f}  median min jitter {s['median_min_jitter']}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
