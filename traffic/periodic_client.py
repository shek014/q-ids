"""Periodic UDP client — the shared mechanism behind BOTH C2 beacons and benign keepalive/polling.

A beacon and a benign heartbeat are the *same* thing on the wire: a host that sends a small packet
to one destination every so often. Each check-in is a single UDP datagram (DNS/heartbeat-style C2,
the clean realistic case), so a flow's packet inter-arrival times ARE the check-in intervals — the
timing signal lives directly in mean_iat / std_iat / cv_iat instead of being buried under TCP
handshake micro-timing. What separates the two roles is not the mechanism but the *statistics*:

  - a naive C2 beacon is rigid        -> low jitter  -> low cv_iat, consistent payload size
  - benign keepalive/polling is loose -> high jitter -> high cv_iat, varied payload size

So this one module generates both roles; the scenario picks the regime via `jitter` / `size_jitter`
(0 = rigid, 1 = ±100%) and the collector labels the flow by the host's role (attack entry -> c2,
benign entry -> benign). With intervals drawn from the SAME range for both, timing regularity and
size consistency are the ONLY things distinguishing them — which is exactly the axis Step 3's
evasion dials. A functional beacon just has to keep checking in, so raising jitter to mimic benign
traffic is realizable and (like the SYN flood's source-port pool) essentially free of functional cost.

Uses an ordinary UDP socket (no root); needs a listener echoing replies (traffic.udp_echo). A
check-in is functional iff the echo reply comes back; the per-run check-in count printed here is the
raw material for Step 3's functional oracle (max tolerable gap between successful check-ins).
"""
import argparse
import random
import socket
import time


def _payload(size):
    body = b"PING " + b"x" * max(0, size - 5)
    return body[:max(1, size)]


def run(target, port, interval=5.0, jitter=0.1, payload_size=64, size_jitter=0.0,
        duration=30, timeout=1.0, seed=None, log=None, verbose=True):
    rng = random.Random(seed)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    logf = open(log, "a") if log else None   # timestamp of each successful (round-trip) check-in
    end = time.time() + duration
    checkins = attempts = 0
    while time.time() < end:
        attempts += 1
        size = int(round(payload_size * (1 + rng.uniform(-size_jitter, size_jitter))))
        try:
            sock.sendto(_payload(size), (target, port))
            sock.recvfrom(1024)              # echo reply confirms the check-in reached the C2/service
            checkins += 1
            if logf:
                logf.write(f"{time.time():.6f}\n")
                logf.flush()
        except OSError:
            pass                             # missed check-in (no reply within timeout)
        time.sleep(max(0.0, interval * (1 + rng.uniform(-jitter, jitter))))
    sock.close()
    if logf:
        logf.close()
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
    parser.add_argument("--log", default=None, help="append each successful check-in's timestamp here")
    args = parser.parse_args()
    run(args.target, args.port, args.interval, args.jitter, args.payload_size,
        args.size_jitter, args.duration, args.timeout, args.seed, args.log)


if __name__ == "__main__":
    main()
