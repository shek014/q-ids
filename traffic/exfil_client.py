"""Low-and-slow data-exfiltration client — the SECOND evasion-amenable attack family (PROTOTYPE).

Mirrors the C2-beacon substrate (traffic.periodic_client) in spirit: the *function* — move a fixed
body of stolen data out to an external collector — is separable from the *traffic shape*: how fast,
in what size bursts. A naive exfil dumps the whole payload as fast as it can, a short, loud,
high-throughput flow (large byte_count, high bytes_per_second) that stands out from the small
interactive traffic around it. But the same bytes can be dribbled out "low and slow": throttled to a
low rate and spread over a long window so bytes_per_second falls into the benign background band.

What separates a naive exfil from a benign bulk transfer (backup / sync) is therefore NOT the
mechanism but the *rate / volume statistics* — exactly the axis the evasion dials:

  - naive exfil -> high rate, short duration -> high bytes_per_second, high packets_per_second (loud)
  - throttled   -> low rate,  long duration  -> low  bytes_per_second                      (benign-like)

Unlike C2 jitter (which is essentially free), throttling carries a real functional COST: the slower
the trickle, the longer the exfil takes, and it must still deliver the WHOLE payload before an
operational deadline. So the functional oracle bites here in a way it did not for C2 — throttle too
hard and the transfer doesn't finish in time (not functional). The evasion's job is to find the
gentlest throttling that drops the flow below the detector's threshold while still completing inside
the deadline.

Uses a plain TCP socket (reliable delivery = clean functional signal: bytes delivered == volume). The
far end is traffic.exfil_sink, which counts received bytes and records completion. --log records the
cumulative bytes delivered and a timestamp per chunk — the raw material for the exfil functional
oracle (full payload delivered, within the deadline). No root needed.
"""
import argparse
import random
import socket
import time


def _chunk(n):
    return b"D" * n


def run(target, port, volume=1_000_000, rate=200_000.0, chunk_size=4096, jitter=0.0,
        deadline=120.0, seed=None, log=None, verbose=True):
    """Exfiltrate `volume` bytes to target:port at ~`rate` bytes/sec, paced by sleeping between
    `chunk_size`-byte sends. `rate` is the evasion dial (lower = stealthier, slower, longer). `jitter`
    randomizes the inter-chunk delay as a fraction. Stops at `deadline` seconds even if incomplete (in
    which case the exfil is non-functional). Returns the number of bytes delivered."""
    rng = random.Random(seed)
    start = time.time()
    end = start + deadline
    sent = 0
    logf = open(log, "a") if log else None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(max(5.0, chunk_size / max(rate, 1.0) + 5.0))
    try:
        sock.connect((target, port))
        while sent < volume and time.time() < end:
            n = min(chunk_size, volume - sent)
            sock.sendall(_chunk(n))
            sent += n
            if logf:
                logf.write(f"{time.time():.6f} {sent}\n")
                logf.flush()
            if sent < volume:
                # pace to the target rate: nominal time for this chunk is n / rate, jittered
                delay = (n / max(rate, 1.0)) * (1 + rng.uniform(-jitter, jitter))
                time.sleep(max(0.0, delay))
        sock.shutdown(socket.SHUT_WR)   # signal end-of-payload; let the sink drain and confirm
    except OSError:
        pass                            # connection refused / reset / timed out -> partial, non-functional
    finally:
        sock.close()
        if logf:
            logf.close()
    complete = sent >= volume
    elapsed = time.time() - start
    if verbose:
        print(f"exfil_client: {sent}/{volume} bytes to {target}:{port} in {elapsed:.1f}s "
              f"(rate={rate:.0f}B/s chunk={chunk_size}b jitter={jitter} "
              f"{'COMPLETE' if complete else 'INCOMPLETE'})")
    return sent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--volume", type=int, default=1_000_000, help="total bytes to exfiltrate")
    parser.add_argument("--rate", type=float, default=200_000.0,
                        help="target throughput in bytes/sec (lower = stealthier; the core evasion dial)")
    parser.add_argument("--chunk-size", type=int, default=4096, help="bytes per send")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="inter-chunk delay randomness as a fraction (0=steady trickle)")
    parser.add_argument("--deadline", type=float, default=120.0,
                        help="functional deadline in seconds; stop even if incomplete")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log", default=None, help="append '<timestamp> <cumulative_bytes>' per chunk here")
    args = parser.parse_args()
    run(args.target, args.port, args.volume, args.rate, args.chunk_size, args.jitter,
        args.deadline, args.seed, args.log)


if __name__ == "__main__":
    main()
