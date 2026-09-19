"""Varied benign client — the key to making benign traffic overlap stealthy attacks.

A realistic host doesn't just do one thing to one port: it opens many short connections across
several services, and some attempts fail (unavailable/misconfigured services). This generator does
exactly that, via ordinary kernel sockets, producing benign flows whose footprint spans the
regions where stealthy attacks live:
  - contacting several distinct ports  -> unique_dst_ports in the low-teens, overlapping small scans
  - failed connections (SYN, no completion) -> SYN-heavy/low-ACK flows, overlapping low-rate floods
Randomizing n_ports / closed_ratio / n_connections per run spreads benign across "clearly normal"
to "looks a bit attack-shaped" — the overlap a meaningful detector comparison and evasion study need.
"""
import argparse
import random
import socket
import time


def run(target, open_ports, closed_ports, n_ports=3, closed_ratio=0.2,
        n_connections=30, timeout=0.4, seed=None):
    rng = random.Random(seed)
    n_closed = min(len(closed_ports), int(round(n_ports * closed_ratio)))
    n_open = max(1, n_ports - n_closed)
    chosen = rng.sample(open_ports, min(n_open, len(open_ports)))
    chosen += rng.sample(closed_ports, n_closed)

    made = 0
    for _ in range(n_connections):
        port = rng.choice(chosen)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((target, port))               # open port: completes; closed: fails/times out
            s.sendall(b"GET / HTTP/1.0\r\n\r\n")
            s.recv(1024)
        except OSError:
            pass                                     # failed attempt -> SYN(s) with no completion
        finally:
            s.close()
        made += 1
        time.sleep(rng.uniform(0.05, 0.4))
    print(f"app_client: {made} connections to {target} across {len(chosen)} ports "
          f"({n_closed} closed)")


def _ports(spec):
    return [int(p) for p in str(spec).split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--open-ports", required=True, help="ports where services listen")
    parser.add_argument("--closed-ports", required=True, help="unused ports (connections fail)")
    parser.add_argument("--n-ports", type=int, default=3, help="distinct ports to contact this run")
    parser.add_argument("--closed-ratio", type=float, default=0.2, help="fraction of ports that are closed")
    parser.add_argument("--n-connections", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=0.4)
    args = parser.parse_args()
    run(args.target, _ports(args.open_ports), _ports(args.closed_ports),
        args.n_ports, args.closed_ratio, args.n_connections, args.timeout)


if __name__ == "__main__":
    main()
