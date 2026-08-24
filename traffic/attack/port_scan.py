"""Reconnaissance scan via Nmap — wraps the nmap binary. Requires root for SYN/stealth
scan types. Only point this at hosts inside the emulated Mininet topology.

Evasive variants use nmap's own timing templates and decoys (see README > Threat model):
  --timing 1        slow-rate scan to blend into the benign statistical envelope
  --decoys RND:5    scan appears to come from 5 additional random decoy sources
"""
import argparse
import subprocess

SCAN_FLAGS = {
    "syn": "-sS",
    "connect": "-sT",
    "fin": "-sF",
    "xmas": "-sX",
    "null": "-sN",
    "udp": "-sU",
}


def run(target, ports="1-1024", scan_type="syn", timing=3, decoys=None, verbose=True):
    cmd = ["nmap", SCAN_FLAGS[scan_type], "-p", ports, "-T", str(timing)]
    if decoys:
        cmd += ["-D", decoys]
    cmd.append(target)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(result.stdout)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ports", default="1-1024")
    parser.add_argument("--scan-type", choices=SCAN_FLAGS.keys(), default="syn")
    parser.add_argument("--timing", type=int, default=3, choices=range(6))
    parser.add_argument("--decoys", default=None, help='e.g. "RND:5" for 5 random decoys')
    args = parser.parse_args()
    run(args.target, args.ports, args.scan_type, args.timing, args.decoys)


if __name__ == "__main__":
    main()
