"""Periodic TCP client — the shared mechanism behind BOTH C2 beacons and benign keepalive/polling.

A beacon and a benign heartbeat are the *same* thing on the wire: a host that opens a short
connection to a service every so often, sends a little, reads a little, and hangs up. What tells
them apart is not the mechanism but the *statistics* of the cadence and payload:

  - a naive C2 beacon is rigid        -> low timing jitter, consistent check-in size
  - benign keepalive/polling is loose -> naturally jittered intervals, varied payloads

So this one module generates both roles; the scenario picks the regime via `jitter` / `size_jitter`
(0 = rigid, 1 = ±100%) and the collector labels the flow by the host's role (attack entry -> c2,
benign entry -> benign). That makes the two classes near-boundary *by construction*: the only thing
separating them is the timing/size distribution — which is exactly the axis Step 3's evasion dials.
A functional beacon just has to keep checking in, so raising jitter to mimic benign traffic is
realizable and (like the SYN flood's source-port pool) essentially free of functional cost.

Uses ordinary kernel sockets, so it needs a listener on the far side (traffic.benign.services) and
no root. The check-in is functional iff the connection completes and a reply is read back; the
per-run check-in count printed here is the raw material for Step 3's functional oracle
(max tolerable gap between successful check-ins).
"""
import argparse
import random
import socket
import time


def _payload(size):
    body = b"PING " + b"x" * max(0, size - 5)
    return body[:max(1, size)]


def run(target, port, interval=5.0, jitter=0.1, payload_size=64, size_jitter=0.0,
        duration=30, timeout=1.0, seed=None, verbose=True):
    rng = random.Random(seed)
    end = time.time() + duration
    checkins = attempts = 0
    while time.time() < end:
        attempts += 1
        size = int(round(payload_size * (1 + rng.uniform(-size_jitter, size_jitter))))
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((target, port))
            s.sendall(_payload(size))
            s.recv(1024)
            checkins += 1                    # completed check-in: reached C2 / service, got a reply
        except OSError:
            pass                             # missed check-in (server busy/unreachable)
        finally:
            s.close()
        time.sleep(max(0.0, interval * (1 + rng.uniform(-jitter, jitter))))
    if verbose:
        print(f"periodic_client: {checkins}/{attempts} check-ins to {target}:{port} "
              f"(interval={interval}s jitter={jitter} size={payload_size}b size_jitter={size_jitter})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between check-ins")
    parser.add_argument("--jitter", type=float, default=0.1,
                        help="interval randomness as a fraction (0=rigid beacon, ~0.5=benign-like)")
    parser.add_argument("--payload-size", type=int, default=64, help="check-in payload bytes")
    parser.add_argument("--size-jitter", type=float, default=0.0,
                        help="payload-size randomness as a fraction (0=consistent, higher=varied)")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args.target, args.port, args.interval, args.jitter, args.payload_size,
        args.size_jitter, args.duration, args.timeout, args.seed)


if __name__ == "__main__":
    main()
