"""Port scan via Scapy — sweeps a SYN (or FIN/XMAS/NULL) probe across a range of destination
ports from the host's real address. Requires root. Only point this at the emulated Mininet hosts.

Scapy rather than nmap: nmap's scans depend on receiving and interpreting the target's replies,
which stalls against Mininet hosts (closed ports don't refuse cleanly, and raw-socket sniffing
hangs in the namespace). A Scapy sweep just emits the probes — all we need to generate detectable
recon traffic: one source hitting many destination ports (high unique_dst_ports, low
unique_src_ports, which is exactly what distinguishes a scan from a spoofed flood).
"""
import argparse
import time

from scapy.all import IP, TCP, RandShort, send

SCAN_FLAGS = {"syn": "S", "fin": "F", "xmas": "FPU", "null": ""}


def _parse_ports(spec):
    ports = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            ports.extend(range(int(a), int(b) + 1))
        else:
            ports.append(int(part))
    return ports


def run(target, ports="1-1024", scan_type="syn", rate=500, verbose=True):
    flags = SCAN_FLAGS[scan_type]
    port_list = _parse_ports(ports)
    sport = int(RandShort())          # one source port for the whole scan -> unique_src_ports = 1
    interval = 1.0 / rate if rate > 0 else 0
    sent = 0
    for port in port_list:
        send(IP(dst=target) / TCP(sport=sport, dport=port, flags=flags), verbose=0)
        sent += 1
        if interval:
            time.sleep(interval)
    if verbose:
        print(f"port_scan: sent {sent} {scan_type} probes to {target} ports {ports}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ports", default="1-1024")
    parser.add_argument("--scan-type", choices=SCAN_FLAGS.keys(), default="syn")
    parser.add_argument("--rate", type=float, default=500, help="probes per second")
    args = parser.parse_args()
    run(args.target, args.ports, args.scan_type, args.rate)


if __name__ == "__main__":
    main()
