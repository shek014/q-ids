"""The evasion query primitive: run ONE real beacon with a given parameter set, in a short Mininet
capture, and return (its extracted flow features, its functional verdict). This is the expensive
oracle the line search wraps — each call is a live capture, so it needs root (see README).

A query is deliberately minimal: two hosts (h1 = C2/echo + capture, h2 = beacon). Flow features are
per source MAC, so the beacon's flow is identical whether or not other hosts are chattering — no need
to reproduce the full training topology here, only the beacon itself. The capture window matches the
training window (60s) so the query flow is in-distribution with the trained detector (the features
duration/packet_count scale with the window; the window-invariant redesign is the planned follow-up).
"""
import argparse
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink

from capture.capture import Capture
from topology.topo import IDSTopo
from features.extract import extract_pcap
from features.dataset import FEATURE_NAMES
from evasion import oracle

C2_PORT = 9999


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
    work.mkdir(parents=True, exist_ok=True)
    pcap_path = work / "query.pcap"
    log_path = work / "checkins.log"
    if log_path.exists():
        log_path.unlink()

    net = Mininet(topo=IDSTopo(n_hosts=2), controller=OVSController, link=TCLink)
    net.start()
    try:
        h1, h2 = net.get("h1"), net.get("h2")
        for h in (h1, h2):
            for intf in h.intfList():
                if intf.name != "lo":
                    h.cmd(f"ethtool -K {intf.name} gso off tso off gro off lro off")
        net.staticArp()

        echo = h1.popen(["python3", "-m", "traffic.udp_echo", "--ports", str(C2_PORT)])
        time.sleep(1.0)  # let the echo server bind

        beacon_cmd = [
            "python3", "-m", "traffic.periodic_client",
            "--target", h1.IP(), "--port", str(C2_PORT),
            "--interval", str(theta["interval"]),
            "--jitter", str(theta["jitter"]),
            "--payload-size", str(int(theta.get("payload_size", 64))),
            "--size-jitter", str(theta.get("size_jitter", 0.0)),
            "--duration", str(window),
            "--seed", str(seed),
            "--log", str(log_path),
        ]
        with Capture(iface="h1-eth0", out_path=str(pcap_path), node=h1):
            t0 = time.time()
            beacon = h2.popen(beacon_cmd)
            time.sleep(window + 1.0)
            if beacon.poll() is None:
                beacon.terminate()
        if echo.poll() is None:
            echo.terminate()

        X, _ = extract_pcap(pcap_path, mac_labels={h2.MAC(): "c2"}, exclude_macs=[h1.MAC()])
        features = X[0] if len(X) else None
        func = oracle.evaluate(oracle.read_checkins(log_path), max_interval,
                               window_start=t0, window_end=t0 + window)
        return QueryResult(theta=dict(theta), features=features, functionality=func, n_flows=len(X))
    finally:
        net.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=float, default=4.0)
    p.add_argument("--jitter", type=float, default=0.1)
    p.add_argument("--size-jitter", type=float, default=0.0)
    p.add_argument("--payload-size", type=int, default=64)
    p.add_argument("--window", type=float, default=60.0)
    p.add_argument("--max-interval", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    theta = {"interval": args.interval, "jitter": args.jitter,
             "size_jitter": args.size_jitter, "payload_size": args.payload_size}
    r = run_query(theta, window=args.window, max_interval=args.max_interval, seed=args.seed)

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
        print("no beacon flow extracted (beacon produced no captured traffic)")


if __name__ == "__main__":
    main()
