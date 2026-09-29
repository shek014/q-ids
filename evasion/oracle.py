"""Functional oracle for the C2-beacon evasion study.

An evasion is only meaningful if the beacon still WORKS after it. "Works" = it maintains its control
channel: successful (round-trip-confirmed) check-ins keep arriving, with no gap longer than the
operator's tolerance. traffic.periodic_client logs a timestamp per successful check-in; this reads
that log and decides functionality.

Deliberately independent of Mininet/scapy (pure stdlib + numpy) so the non-sudo evasion search loop
can call it on a log the sudo capture produced.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Functionality:
    functional: bool
    n_checkins: int
    max_gap: float       # longest interval (s) with no successful check-in; inf if too few check-ins
    max_interval: float  # the tolerance it was judged against


def read_checkins(path):
    """Read successful-check-in timestamps written by periodic_client --log. Missing/empty -> []."""
    try:
        with open(path) as f:
            return sorted(float(line) for line in f if line.strip())
    except FileNotFoundError:
        return []


def evaluate(checkin_times, max_interval, window_start=None, window_end=None, min_checkins=2):
    """A beacon is functional iff it made >= min_checkins successful check-ins AND no gap between
    consecutive ones (optionally including the window edges) exceeds max_interval. Passing
    window_start/window_end catches a beacon that falls silent partway through the capture."""
    times = sorted(checkin_times)
    n = len(times)
    if n < min_checkins:
        return Functionality(False, n, float("inf"), max_interval)

    edges = ([window_start] if window_start is not None else []) + times + \
            ([window_end] if window_end is not None else [])
    max_gap = float(np.diff(edges).max())
    return Functionality(max_gap <= max_interval, n, max_gap, max_interval)
