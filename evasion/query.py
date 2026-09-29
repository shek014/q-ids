"""The evasion query primitive: run ONE real beacon with a given parameter set and return
(its extracted flow features, its functional verdict). This is the expensive oracle the line
search wraps — each call is a live capture.

Runs in the .venv (numpy/scapy/models), the same environment that trained the detectors, so no
version skew touches the PCA-based detectors. The Mininet capture itself lives in system python3
(no numpy there), so it is delegated to evasion/capture_helper.py as a subprocess. Drive the whole
thing as root:  sudo .venv/bin/python -m evasion.run ...  (root is inherited by the helper, which
needs it for Mininet). A query is deliberately minimal: two hosts (h1 = C2/echo + capture, h2 =
beacon). Flow features are per source MAC, so the beacon's flow is the same with or without other
hosts chattering. The 60s window matches training so the query flow is in-distribution.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from features.extract import extract_pcap
from features.dataset import FEATURE_NAMES
from evasion import oracle

# system python3 has Mininet (the .venv does not); use it explicitly for the capture half
SYSTEM_PYTHON = "/usr/bin/python3"


@dataclass
class QueryResult:
    theta: dict
    features: object                    # np.ndarray (len FEATURE_NAMES) or None if no flow extracted
    functionality: oracle.Functionality
    n_flows: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def evaded_candidate(self):
        return self.features is not None and self.functionality.functional


def run_query(theta, window=60.0, max_interval=30.0, seed=0, work_dir=None):
    """theta: {interval, jitter, size_jitter, payload_size?}. Returns a QueryResult."""
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="evq_"))
    cmd = [
        SYSTEM_PYTHON, "-m", "evasion.capture_helper",
        "--interval", str(theta["interval"]),
        "--jitter", str(theta["jitter"]),
        "--size-jitter", str(theta.get("size_jitter", 0.0)),
        "--payload-size", str(int(theta.get("payload_size", 64))),
        "--window", str(window),
        "--seed", str(seed),
        "--out-dir", str(work),
    ]
    subprocess.run(cmd, check=True)  # inherits root from the driver; builds+captures the topology

    meta = json.loads((work / "meta.json").read_text())
    X, _ = extract_pcap(work / "query.pcap", mac_labels={meta["beacon_mac"]: "c2"},
                        exclude_macs=[meta["capture_mac"]])
    features = X[0] if len(X) else None
    func = oracle.evaluate(oracle.read_checkins(work / "checkins.log"), max_interval,
                           window_start=meta["window_start"],
                           window_end=meta["window_start"] + meta["window"])
    return QueryResult(theta=dict(theta), features=features, functionality=func, n_flows=len(X))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=float, default=4.0)
    p.add_argument("--jitter", type=float, default=0.1)
    p.add_argument("--size-jitter", type=float, default=0.0)
    p.add_argument("--payload-size", type=int, default=64)
    p.add_argument("--window", type=float, default=60.0)
    p.add_argument("--max-interval", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    theta = {"interval": a.interval, "jitter": a.jitter,
             "size_jitter": a.size_jitter, "payload_size": a.payload_size}
    r = run_query(theta, window=a.window, max_interval=a.max_interval, seed=a.seed)

    f = r.functionality
    print(f"\ntheta={theta}")
    print(f"flows extracted: {r.n_flows}")
    print(f"functional: {f.functional}  (check-ins={f.n_checkins}  max_gap={f.max_gap:.2f}s  "
          f"tolerance={f.max_interval:.0f}s)")
    if r.features is not None:
        print("features:")
        for name, val in zip(FEATURE_NAMES, r.features):
            print(f"  {name:<20}{val:.4f}")
    else:
        print("no beacon flow extracted", file=sys.stderr)


if __name__ == "__main__":
    main()
