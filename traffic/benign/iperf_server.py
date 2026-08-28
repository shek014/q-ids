"""Long-running iperf3 server, started on the target host before benign iperf clients
connect (see topology/run.py). Runs until terminated by the scenario runner. Requires
iperf3 on the host — Mininet host images typically ship with it.
"""
import argparse
import subprocess


def run(port=5201):
    # -s server, -1 would exit after one client; we stay up for the whole scenario window.
    subprocess.run(["iperf3", "-s", "-p", str(port)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5201)
    args = parser.parse_args()
    run(args.port)


if __name__ == "__main__":
    main()
