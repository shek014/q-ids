"""Line search on jitter: the core of the evasion study. For one beacon instance (fixed seed,
fixed interval/size), sweep timing jitter upward and find the SMALLEST jitter at which the detector
flips the beacon to 'benign' while it stays functional. That minimal evading jitter is the cost of
evasion; the number of queries spent reaching it is queries-to-evasion.

The search is detector- and query-agnostic: it takes a `prob_benign(features) -> float` callable and
a `query_fn(theta) -> QueryResult`, so the same loop drives the MLP or the VQC, real captures or a
stub in tests.
"""
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    seed: int
    base_theta: dict
    evaded: bool
    min_jitter: float          # smallest jitter that evaded while functional; inf if never
    queries: int               # detector queries spent (captures run)
    trajectory: list = field(default_factory=list)  # [(jitter, prob_benign|None, functional)]


def line_search(prob_benign, query_fn, base_theta, values, seed=0,
                threshold=0.5, early_stop=True, axis="jitter"):
    """values: ascending list to try along `axis` (jitter, or size_jitter for the step-4 escalation).
    A query "evades" when the detector's P(benign) >= threshold AND the beacon is still functional.
    With early_stop, returns at the first evading value (the minimal one, since values ascend);
    otherwise runs the whole grid for a full P(benign)-vs-value curve. min_jitter holds the minimal
    evading value of whichever axis was searched."""
    trajectory = []
    evaded, min_jitter = False, float("inf")

    for j in values:
        theta = {**base_theta, axis: j}
        result = query_fn(theta, seed)
        if result.features is None:
            trajectory.append((j, None, result.functionality.functional))
            continue

        p = float(prob_benign(result.features)[0])
        functional = result.functionality.functional
        trajectory.append((j, p, functional))

        if p >= threshold and functional and not evaded:
            evaded, min_jitter = True, j
            if early_stop:
                break

    return SearchResult(seed=seed, base_theta=dict(base_theta), evaded=evaded,
                        min_jitter=min_jitter, queries=len(trajectory), trajectory=trajectory)
