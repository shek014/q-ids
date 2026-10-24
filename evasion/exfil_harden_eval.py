"""Does adversarial training defeat the throttle evasion? Scores the HELD-OUT sweep seeds (never seen
in hardening) against each EXFIL detector before and after hardening, offline (no captures, no root).
Mirrors evasion/harden_eval.py (the C2 version):

    .venv/bin/python -m evasion.exfil_harden_eval --sweep results/evasion/exfil_sweep.npz --heldout 10-14
"""
import argparse

from evasion.detector import Detector
from evasion.exfil_transfer import min_evading_throttle, _stats
from evasion.harden import split_instances
import numpy as np


def slice_sweep(sweep, seeds):
    return {"features": sweep["features"][seeds], "functional": sweep["functional"][seeds],
            "got": sweep["got"][seeds], "throttles": sweep["throttles"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", default="results/evasion/exfil_sweep.npz")
    p.add_argument("--data", default="data/exfil_dataset.npz")
    p.add_argument("--train-frac", type=float, default=0.67,
                   help="must match evasion.harden; the complementary held-out instances are evaluated")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    sweep = np.load(args.sweep)
    _, heldout_idx = split_instances(sweep, args.train_frac)
    sub = slice_sweep(sweep, heldout_idx)
    print(f"held-out instances {heldout_idx}: {sub['features'].shape[0]} exfils "
          f"x {sub['features'].shape[1]} throttle levels (throttle-only evasions)\n")

    print(f"{'model':38s}{'evasion rate':>14s}{'median throttle':>18s}")
    for orig, hard in [("exfil_classical_native", "exfil_classical_native_hardened"),
                       ("exfil_classical_matched", "exfil_classical_matched_hardened"),
                       ("exfil_quantum", "exfil_quantum_hardened")]:
        for d in (orig, hard):
            det = Detector.from_result(f"results/{d}", args.data)
            ev, mj, _ = min_evading_throttle(det, sub, args.threshold)
            s = _stats(ev, mj)
            mj_str = f"{s['median_min_throttle']}" if s['median_min_throttle'] is not None else "-"
            print(f"{d:38s}{s['evasion_rate']:>14.2f}{mj_str:>18s}")
        print()


if __name__ == "__main__":
    main()
