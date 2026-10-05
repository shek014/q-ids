"""Capture-only half of the EXFIL evasion query (PROTOTYPE), run under SYSTEM python3 (which has
Mininet but no numpy). Builds the minimal 2-host topology, runs ONE data-exfiltration transfer with
the given parameters to an exfil_sink, captures the traffic, and writes the pcap + a meta sidecar +
the delivery-progress log. It does NO feature extraction (that needs numpy/scapy and happens in the
.venv half, evasion/exfil_query.py), so this file imports nothing beyond Mininet + the pure-stdlib
capture wrapper.

Mirrors evasion/capture_helper.py (the C2 version). Invoked as a subprocess by evasion/exfil_query.py;
not meant to be run by hand.
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

EXFIL_PORT = 8443


def capture(theta, window, seed, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcap_path = out / "query.pcap"
    log_path = out / "progress.log"
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

        sink = h1.popen(["python3", "-m", "traffic.exfil_sink", "--ports", str(EXFIL_PORT)])
        time.sleep(1.0)

        exfil_cmd = [
            "python3", "-m", "traffic.exfil_client",
            "--target", h1.IP(), "--port", str(EXFIL_PORT),
            "--volume", str(int(theta["volume"])),
            "--rate", str(theta["rate"]),                        # already the EFFECTIVE (throttled) rate
            "--chunk-size", str(int(theta.get("chunk_size", 4096))),
            "--jitter", str(theta.get("jitter", 0.0)),
            "--deadline", str(theta.get("deadline", window)),
            "--seed", str(seed),
            "--log", str(log_path),
        ]
        with Capture(iface="h1-eth0", out_path=str(pcap_path), node=h1):
            t0 = time.time()
            client = h2.popen(exfil_cmd)
            time.sleep(window + 1.0)
            if client.poll() is None:
                client.terminate()
        if sink.poll() is None:
            sink.terminate()

        (out / "meta.json").write_text(json.dumps({
            "exfil_mac": h2.MAC(), "capture_mac": h1.MAC(),
            "window_start": t0, "window": window, "theta": theta, "seed": seed,
        }))
    finally:
        net.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", type=int, required=True)
    p.add_argument("--rate", type=float, required=True, help="effective (post-throttle) bytes/sec")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--window", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    capture({"volume": a.volume, "rate": a.rate, "chunk_size": a.chunk_size,
             "jitter": a.jitter, "deadline": a.deadline}, a.window, a.seed, a.out_dir)


if __name__ == "__main__":
    main()
