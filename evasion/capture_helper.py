"""Capture-only half of the evasion query, run under SYSTEM python3 (which has Mininet but no
numpy). Builds the minimal 2-host topology, runs one beacon with the given parameters against a
UDP echo C2, captures the traffic, and writes the pcap + a meta sidecar + the check-in log. It does
NO feature extraction (that needs numpy/scapy and happens in the .venv half, evasion/query.py), so
this file imports nothing beyond Mininet + the pure-stdlib capture wrapper.

Invoked as a subprocess by evasion/query.py; not meant to be run by hand.
"""
import argparse
import json
import time
from pathlib import Path

from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink

from capture.capture import Capture
from topology.topo import IDSTopo

C2_PORT = 9999


def capture(theta, window, seed, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcap_path = out / "query.pcap"
    log_path = out / "checkins.log"
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
        time.sleep(1.0)

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

        (out / "meta.json").write_text(json.dumps({
            "beacon_mac": h2.MAC(), "capture_mac": h1.MAC(),
            "window_start": t0, "window": window, "theta": theta, "seed": seed,
        }))
    finally:
        net.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=float, required=True)
    p.add_argument("--jitter", type=float, required=True)
    p.add_argument("--size-jitter", type=float, default=0.0)
    p.add_argument("--payload-size", type=int, default=64)
    p.add_argument("--window", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    capture({"interval": a.interval, "jitter": a.jitter, "size_jitter": a.size_jitter,
             "payload_size": a.payload_size}, a.window, a.seed, a.out_dir)


if __name__ == "__main__":
    main()
