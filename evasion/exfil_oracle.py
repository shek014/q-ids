"""Functional oracle for the data-exfiltration evasion study (PROTOTYPE).

An exfil evasion is only meaningful if the payload still gets OUT. "Works" here = the full volume was
delivered to the collector within the operator's deadline. traffic.exfil_client logs
'<timestamp> <cumulative_bytes>' per chunk as it sends, and traffic.exfil_sink logs the per-connection
total it actually received; this reads the client's cumulative progress to decide functionality, and
the sink total can cross-check actual delivery (TCP guarantees the sent bytes arrive if the connection
completes).

The contrast with the C2 oracle is the point of this attack family: raising C2 jitter is essentially
free, but throttling to evade here trades directly against this oracle — drop the rate too far and the
transfer does not reach `volume` before the deadline, so the evasion is inadmissible. The functional
constraint therefore bounds how far the evasion can throttle, which is the whole tension of the study.

Deliberately independent of Mininet/scapy (pure stdlib) so the non-sudo evasion search loop can call
it on a log the sudo capture produced.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExfilFunctionality:
    functional: bool
    bytes_delivered: int
    volume: int
    completion_time: Optional[float]  # seconds from window_start to full delivery; None if never completed
    deadline: float


def read_progress(path):
    """Read '<timestamp> <cumulative_bytes>' lines (exfil_client/exfil_sink --log). Lines that don't
    parse are skipped; a missing/empty file yields []. Returns a list of (timestamp, cumulative_bytes)
    sorted by timestamp."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    out.append((float(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    except FileNotFoundError:
        return []
    return sorted(out)


def evaluate(progress, volume, deadline, window_start=None):
    """An exfil is functional iff the cumulative bytes delivered reached `volume` at some logged time
    whose offset from window_start is <= deadline. progress is a list of (timestamp, cumulative_bytes)
    as written by exfil_client --log. If window_start is None the first logged timestamp is used."""
    if not progress:
        return ExfilFunctionality(False, 0, int(volume), None, float(deadline))

    start = window_start if window_start is not None else progress[0][0]
    delivered = max(b for _, b in progress)
    completion_time = None
    for ts, b in progress:
        if b >= volume:
            completion_time = ts - start
            break

    functional = completion_time is not None and completion_time <= deadline
    return ExfilFunctionality(functional, int(delivered), int(volume), completion_time, float(deadline))
