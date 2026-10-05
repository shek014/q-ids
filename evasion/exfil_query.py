"""The EXFIL evasion query primitive (PROTOTYPE): run ONE real exfiltration transfer with a given
parameter set and return (its extracted flow features, its functional verdict). This is the expensive
oracle the line search wraps — each call is a live capture. Mirrors evasion/query.py (the C2 version).

The evasion dials THROTTLE. theta carries a base `rate` (the naive dump rate) and a `throttle` factor;
the effective transfer rate is base_rate / (1 + throttle), so throttle=0 is the naive loud dump and
larger throttle is slower and stealthier — until it is so slow the payload can't finish before the
deadline, at which point the functional oracle rejects it. That tension (stealth vs. completion) is
the whole point of this attack family and the reason its oracle bites where the C2 one did not.

Runs in the .venv (numpy/scapy/models), the same environment that trained the detectors. The Mininet
capture itself lives in system python3 (no numpy there), so it is delegated to
evasion/exfil_capture_helper.py as a subprocess. Drive the whole thing as root:
    sudo .venv/bin/python -m evasion.exfil_run ...
A query is deliberately minimal: two hosts (h1 = exfil collector + capture, h2 = exfil client). Flow
features are per source MAC, so the exfil flow is the same with or without other hosts chattering.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from features.extract import extract_pcap
from features.dataset import FEATURE_NAMES
from evasion import exfil_oracle
from evasion.query import QueryResult   # reuse the generic (theta, features, functionality, n_flows) record

# system python3 has Mininet (the .venv does not); use it explicitly for the capture half
SYSTEM_PYTHON = "/usr/bin/python3"


def effective_rate(theta):
    """Naive base rate throttled by the evasion dial: throttle=0 -> base, larger -> slower."""
    return float(theta["rate"]) / (1.0 + float(theta.get("throttle", 0.0)))


def run_query(theta, window=65.0, seed=0, work_dir=None):
    """theta: {volume, rate (naive base), throttle, chunk_size?, jitter?, deadline?}. Returns a
    QueryResult whose .functionality is an exfil_oracle.ExfilFunctionality (also has .functional, so
    the generic line_search treats it exactly like the C2 one)."""
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="exfilq_"))
    deadline = float(theta.get("deadline", window))
    rate = effective_rate(theta)
    cmd = [
        SYSTEM_PYTHON, "-m", "evasion.exfil_capture_helper",
        "--volume", str(int(theta["volume"])),
        "--rate", str(rate),
        "--chunk-size", str(int(theta.get("chunk_size", 4096))),
        "--jitter", str(theta.get("jitter", 0.0)),
        "--deadline", str(deadline),
        "--window", str(window),
        "--seed", str(seed),
        "--out-dir", str(work),
    ]
    subprocess.run(cmd, check=True)  # inherits root from the driver; builds+captures the topology

    meta = json.loads((work / "meta.json").read_text())
    X, _ = extract_pcap(work / "query.pcap", mac_labels={meta["exfil_mac"]: "exfil"},
                        exclude_macs=[meta["capture_mac"]], class_names=["benign", "exfil"])
    features = X[0] if len(X) else None
    func = exfil_oracle.evaluate(exfil_oracle.read_progress(work / "progress.log"),
                                 volume=int(theta["volume"]), deadline=deadline,
                                 window_start=meta["window_start"])
    return QueryResult(theta=dict(theta), features=features, functionality=func, n_flows=len(X))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", type=int, default=1_200_000)
    p.add_argument("--rate", type=float, default=600_000.0, help="naive base rate (pre-throttle)")
    p.add_argument("--throttle", type=float, default=0.0, help="rate = base / (1 + throttle)")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--jitter", type=float, default=0.1)
    p.add_argument("--deadline", type=float, default=60.0)
    p.add_argument("--window", type=float, default=65.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    theta = {"volume": a.volume, "rate": a.rate, "throttle": a.throttle,
             "chunk_size": a.chunk_size, "jitter": a.jitter, "deadline": a.deadline}
    r = run_query(theta, window=a.window, seed=a.seed)

    f = r.functionality
    print(f"\ntheta={theta}")
    print(f"effective rate: {effective_rate(theta):.0f} B/s   flows extracted: {r.n_flows}")
    print(f"functional: {f.functional}  (delivered={f.bytes_delivered}/{f.volume}  "
          f"completion_time={f.completion_time}  deadline={f.deadline:.0f}s)")
    if r.features is not None:
        print("features:")
        for name, val in zip(FEATURE_NAMES, r.features):
            print(f"  {name:<20}{val:.4f}")
    else:
        print("no exfil flow extracted", file=sys.stderr)


if __name__ == "__main__":
    main()
