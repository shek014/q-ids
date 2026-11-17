"""RITA-style beacon scoring for one (source, destination) pair, written from the scoring rules of RITA
v5.1.2 (Active Countermeasures, https://github.com/activecm/rita, analysis/beacons.go; identical in
v5.1.3). This is an independent re-implementation of the published formulas, not a copy of RITA's code.

RITA scores a host pair from its connection log with four subscores weighted 0.25 each by default:
timestamp, data size, histogram and duration. The histogram and duration subscores are built from a
24-hour connection histogram with hourly bins, and the duration subscore is zero unless the pair was
seen in at least 6 distinct hours (default_config.hjson), so neither carries information over the
10-minute runs used here. We therefore implement the two per-connection subscores and report their
mean, which is RITA's weighted score with the two time-of-day subscores removed and the weights
renormalised.

  timestamp subscore  connection start times truncated to whole seconds (RITA stores them as uint32),
                      sorted; deltas between consecutive times; at least 4 timestamps and 3 non-zero
                      deltas are required; zero deltas are dropped; score = statistical_score(deltas,
                      default 1)
  data-size subscore  bytes sent by the source per connection (RITA: src_ip_bytes), at least 3
                      connections; score = statistical_score(sizes, default 0)
  statistical_score   mean of the Bowley-skewness score and the MAD score, rounded to 3 decimals
  Bowley skewness     (Q1 + Q3 - 2 Q2) / (Q3 - Q1), set to 0 when Q3 - Q1 < 10 or Q2 equals Q1 or Q3;
                      score = 1 - |skew|. Quartiles split the sorted data in halves (the median is left
                      out of both halves for odd n), as in github.com/montanaflynn/stats v0.7.1
  MAD score           (median - MAD) / median when median >= 1, else the default; clipped at 0

RITA ignores host pairs with fewer than 4 connections (unique_connection_threshold); they score 0 here.
"""
import math

import numpy as np

MIN_CONNECTIONS = 4


def _round3(x):
    return math.floor(x * 1000 + 0.5) / 1000          # Go math.Round (half away from zero), x >= 0 here


def quartiles(sorted_x):
    n = len(sorted_x)
    c1, c2 = (n // 2, n // 2) if n % 2 == 0 else ((n - 1) // 2, (n - 1) // 2 + 1)
    return float(np.median(sorted_x[:c1])), float(np.median(sorted_x)), float(np.median(sorted_x[c2:]))


def bowley_score(sorted_x):
    q1, q2, q3 = quartiles(sorted_x)
    den = q3 - q1
    skew = (q1 + q3 - 2 * q2) / den if den >= 10 and q2 != q1 and q2 != q3 else 0.0
    return 1.0 - abs(skew)


def mad_score(sorted_x, default):
    m = float(np.median(sorted_x))
    mad = float(np.median(np.abs(np.asarray(sorted_x) - m)))
    score = (m - mad) / m if m >= 1 else default
    return 0.0 if (math.isnan(score) or score < 0) else score


def statistical_score(values, default):
    x = np.sort(np.asarray(values, float))
    return _round3((bowley_score(x) + mad_score(x, default)) / 2.0)


def timestamp_score(ts):
    """ts: connection start times in seconds (epoch or any common origin)."""
    t = np.sort(np.floor(np.asarray(ts, float)))
    if len(t) < 4:
        return 0.0
    d = np.sort(np.diff(t))
    d = d[d > 0]
    if len(d) < 3:
        return 0.0
    return statistical_score(d, 1.0)


def datasize_score(sizes):
    if len(sizes) < 3:
        return 0.0
    return statistical_score(sizes, 0.0)


def beacon_score(ts, sizes):
    """Returns (score, timestamp_subscore, datasize_subscore); score is their mean."""
    if len(ts) < MIN_CONNECTIONS:
        return 0.0, 0.0, 0.0
    a, b = timestamp_score(ts), datasize_score(sizes)
    return _round3((a + b) / 2.0), a, b
