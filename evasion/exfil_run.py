"""EXFIL evasion experiment driver (PROTOTYPE). For a trained benign-vs-exfil detector, runs the
throttle line search over many exfil instances (seeds) and reports the headline metrics: evasion
rate, queries-to-evasion, and the minimal evading throttle (the cost of evasion — how far the attacker
must slow the transfer). Each query is a live capture, so run as root:

    sudo .venv/bin/python -m evasion.exfil_run --detector results/exfil_classical_native --seeds 15 \
         --data data/exfil_dataset.npz

Mirrors evasion/run.py (the C2 driver). Unlike the C2 jitter search, throttling trades against the
functional deadline, so a seed can FAIL to evade (the throttle needed to beat the detector would stop
the payload finishing in time). Results are written to results/evasion/<detector>.json.
"""
import argparse
import json
import statistics
from pathlib import Path

from evasion.detector import Detector
from evasion.exfil_query import run_query, effective_rate
from evasion.query import QueryResult
from evasion.exfil_oracle import ExfilFunctionality
from evasion.search import line_search

# a broken capture shouldn't abort a multi-hour run; treat it as a non-evading, non-functional query
_FAILED = QueryResult(theta={}, features=None,
                      functionality=ExfilFunctionality(False, 0, 0, None, 0.0))


def _errored(r):
    """A seed errored (infrastructure, not the detector) if no query produced a flow to score."""
    return bool(r.trajectory) and all(p is None for _, p, _ in r.trajectory)


def summarize(results):
    scored = [r for r in results if not _errored(r)]     # only seeds with real captures count
    evaded = [r for r in scored if r.evaded]
    n = len(scored)
    return {
        "n_exfil": len(results),
        "n_errored": len(results) - len(scored),
        "evasion_rate": len(evaded) / n if n else 0.0,
        "median_queries_to_evasion": statistics.median([r.queries for r in evaded]) if evaded else None,
        # SearchResult.min_jitter holds the minimal evading value of whatever axis was searched; here
        # that axis is throttle, so we surface it under its real name.
        "median_min_throttle": statistics.median([r.min_jitter for r in evaded]) if evaded else None,
        "per_seed": [{"seed": r.seed, "evaded": r.evaded, "errored": _errored(r),
                      "min_throttle": r.min_jitter, "queries": r.queries, "trajectory": r.trajectory}
                     for r in results],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector", default="results/exfil_classical_native", help="a trained result dir")
    p.add_argument("--data", default="data/exfil_dataset.npz")
    p.add_argument("--seeds", type=int, default=15, help="number of exfil instances")
    p.add_argument("--volume", type=int, default=1_200_000, help="fixed payload size (bytes)")
    p.add_argument("--rate", type=float, default=600_000.0, help="fixed naive base rate (pre-throttle)")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--jitter", type=float, default=0.1, help="held inter-chunk jitter")
    p.add_argument("--deadline", type=float, default=60.0, help="operational completion deadline (s)")
    p.add_argument("--throttles", default="0.0,0.5,1.0,2.0,3.0,5.0,8.0,12.0,20.0,30.0",
                   help="ascending throttle grid; effective rate = base/(1+throttle)")
    p.add_argument("--window", type=float, default=65.0, help="capture window (>= deadline)")
    p.add_argument("--threshold", type=float, default=0.5, help="P(benign) to count as evaded")
    p.add_argument("--full-grid", action="store_true", help="evaluate the whole grid (curve), no early stop")
    p.add_argument("--work-dir", default="/tmp/exfil_evasion_work")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    detector = Detector.from_result(args.detector, args.data)
    values = [float(x) for x in args.throttles.split(",")]
    base_theta = {"volume": args.volume, "rate": args.rate, "throttle": 0.0,
                  "chunk_size": args.chunk_size, "jitter": args.jitter, "deadline": args.deadline}

    def query_fn(theta, seed):
        try:
            return run_query(theta, window=args.window, seed=seed, work_dir=args.work_dir)
        except Exception as e:  # noqa: BLE001 - one flaky capture must not kill the batch
            print(f"  query failed (seed={seed}, throttle={theta.get('throttle')}): {e}")
            return _FAILED

    print(f"detector '{detector.name}' ({detector.meta['model']}, {detector.meta['input_mode']})  "
          f"clean test acc {detector.meta['test_accuracy']:.3f}")

    # Preflight: one real capture up front at the naive (throttle=0) setting. Catches infrastructure
    # failure (OVS down, etc.) in ~1 minute instead of a bogus 0.00 evasion rate an hour later.
    print("preflight capture (naive, throttle=0)...")
    try:
        pf = run_query({**base_theta, "throttle": values[0]}, window=args.window,
                       seed=0, work_dir=args.work_dir)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"preflight capture FAILED: {e}\n"
                         "Is OVS up? try: sudo systemctl start ovsdb-server ovs-vswitchd")
    if pf.features is None:
        raise SystemExit("preflight produced no exfil flow — check OVS / capture setup before the run")
    print(f"preflight ok: functional={pf.functionality.functional}, "
          f"P(benign)={detector.prob_benign(pf.features)[0]:.3f}, "
          f"effective_rate={effective_rate({**base_theta, 'throttle': values[0]}):.0f} B/s\n")
    print(f"line search on throttle: volume={args.volume}B, base_rate={args.rate:.0f}B/s, "
          f"deadline={args.deadline:.0f}s, throttles={values}, {args.seeds} exfils\n")

    results = []
    for seed in range(args.seeds):
        r = line_search(detector.prob_benign, query_fn, base_theta, values, seed=seed,
                        threshold=args.threshold, early_stop=not args.full_grid, axis="throttle")
        if _errored(r):
            status = "ERRORED (no capture)"
        elif r.evaded:
            status = f"EVADED  min_throttle={r.min_jitter}"
        else:
            status = "held   (detector held, or no functional evading throttle)"
        print(f"  seed {seed:2d}: {status}  ({r.queries} queries)")
        results.append(r)

    summary = summarize(results)
    out_path = Path(args.out) if args.out else Path("results/evasion") / f"{detector.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"detector": detector.name, "meta": detector.meta,
                                    "config": vars(args), "summary": summary}, indent=2))

    print(f"\nevasion rate: {summary['evasion_rate']:.2f}  "
          f"median queries-to-evasion: {summary['median_queries_to_evasion']}  "
          f"median min throttle: {summary['median_min_throttle']}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
