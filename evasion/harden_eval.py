"""Step 3 of the defense round: does adversarial training defeat the original (jitter-only) evasion?
Scores the HELD-OUT sweep seeds (never seen in hardening) against each model before and after
hardening, offline (no captures, no root):

    .venv/bin/python -m evasion.harden_eval --sweep results/evasion/sweep.npz --heldout 10-14
"""
import argparse

import numpy as np

from evasion.detector import Detector
from evasion.transfer import min_evading_jitter, _stats
from evasion.harden import parse_seeds


def slice_sweep(sweep, seeds):
    return {"features": sweep["features"][seeds], "functional": sweep["functional"][seeds],
            "got": sweep["got"][seeds], "jitters": sweep["jitters"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/sweep.npz")
    p.add_argument("--heldout", default="10-14")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    sweep = np.load(args.sweep)
    sub = slice_sweep(sweep, parse_seeds(args.heldout))
    print(f"held-out seeds {args.heldout}: {sub['features'].shape[0]} beacons "
          f"x {sub['features'].shape[1]} jitter levels (jitter-only evasions)\n")

    print(f"{'model':30s}{'evasion rate':>14s}{'median jitter':>16s}")
    for orig, hard in [("classical_native", "classical_native_hardened"),
                       ("quantum", "quantum_hardened")]:
        for d in (orig, hard):
            det = Detector.from_result(f"results/{d}")
            ev, mj, _ = min_evading_jitter(det, sub, args.threshold)
            s = _stats(ev, mj)
            mj_str = f"{s['median_min_jitter']}" if s['median_min_jitter'] is not None else "-"
            print(f"{d:30s}{s['evasion_rate']:>14.2f}{mj_str:>16s}")
        print()


if __name__ == "__main__":
    main()
