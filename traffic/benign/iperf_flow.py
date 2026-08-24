"""Benign bulk TCP traffic via iperf3 — a normal high-throughput flow (e.g. a file
transfer or backup job). Requires an iperf3 server already running on the target host
(`iperf3 -s`) — Mininet host images typically ship with iperf3 installed.
"""
import argparse
import subprocess


def run(target, duration=10, port=5201, verbose=True):
    cmd = ["iperf3", "-c", target, "-p", str(port), "-t", str(duration)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(result.stdout)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--port", type=int, default=5201)
    args = parser.parse_args()
    run(args.target, args.duration, args.port)


if __name__ == "__main__":
    main()
