"""Benign ICMP echo traffic via Scapy — a normal ping, not a flood. Requires root."""
import argparse
import time

from scapy.all import ICMP, IP, send


def run(target, count=10, interval=1.0, verbose=True):
    for _ in range(count):
        send(IP(dst=target) / ICMP(), verbose=0)
        time.sleep(interval)
    if verbose:
        print(f"ping_flow: sent {count} echo requests to {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    run(args.target, args.count, args.interval)


if __name__ == "__main__":
    main()
