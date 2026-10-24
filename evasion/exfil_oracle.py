"""Functional oracle for the data-exfiltration evasion study (PROTOTYPE).

An exfil evasion is only meaningful if the payload still gets OUT. "Works" here = the full volume was
delivered to the collector within the operator's deadline. traffic.exfil_client logs
'<timestamp> <cumulative_bytes>' per chunk as it sends, and traffic.exfil_sink logs the per-connection
total it actually received; this reads the client's cumulative progress for completion TIMING and, when
the sink log is present, REQUIRES the collector to have received the full volume before calling the
exfil functional (so a transfer the client "sent" but that never fully landed is correctly inadmissible).

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
    bytes_delivered_sink: Optional[int] = None  # collector-side received total; None if no sink log wired
    delivery_confirmed: Optional[bool] = None   # sink received >= volume; None = unknown (client-only)


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


def _sink_total(sink_progress):
    """Collector-side received bytes. exfil_sink --log writes one '<ts> <bytes_received>' line per
    connection, so the delivered total is their sum (a single connection in the normal case). Returns
    None only when no sink log was provided (sink_progress is None) — an empty list means the sink was
    wired but recorded nothing, which counts as zero delivered (and so fails confirmation)."""
    if sink_progress is None:
        return None
    return sum(b for _, b in sink_progress)


def evaluate(progress, volume, deadline, window_start=None, sink_progress=None):
    """An exfil is functional iff the full `volume` was delivered within `deadline`. The client log
    (`progress`, cumulative sent bytes) drives completion TIMING: the offset from window_start to the
    first logged time the cumulative reaches `volume`. When `sink_progress` (the collector's
    per-connection received totals, from exfil_sink --log) is given, delivery must ALSO be confirmed
    receiver-side — the collector must have actually received the full volume, not merely the client
    having called sendall (which only hands bytes to the local TCP buffer). If sink_progress is None the
    verdict falls back to client-side only (delivery_confirmed=None), for captures taken before the sink
    log was wired."""
    sink_total = _sink_total(sink_progress)
    confirmed = None if sink_total is None else sink_total >= volume

    if not progress:
        return ExfilFunctionality(False, 0, int(volume), None, float(deadline),
                                  bytes_delivered_sink=sink_total, delivery_confirmed=confirmed)

    start = window_start if window_start is not None else progress[0][0]
    delivered = max(b for _, b in progress)
    completion_time = None
    for ts, b in progress:
        if b >= volume:
            completion_time = ts - start
            break

    client_ok = completion_time is not None and completion_time <= deadline
    functional = client_ok if confirmed is None else (client_ok and confirmed)
    return ExfilFunctionality(functional, int(delivered), int(volume), completion_time, float(deadline),
                              bytes_delivered_sink=sink_total, delivery_confirmed=confirmed)
