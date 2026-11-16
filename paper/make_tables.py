"""Recompute every results table in paper/tex/main.tex from the saved models and sweeps, offline.

    .venv/bin/python paper/make_tables.py            # prints the tables
    .venv/bin/python paper/make_tables.py --json results/evasion/paper_tables.json

Conventions are in evasion/paper_metrics.py: an attack counts only if the detector flagged it
unmodified, rates carry Wilson 95% intervals, and minimal evading values are (low, high] brackets.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evasion import paper_metrics as pm
from evasion.detector import Detector
from evasion.harden import split_instances

C2 = [("MLP (native)", "classical_native"), ("MLP (matched)", "classical_matched"), ("VQC", "quantum"),
      ("Random Forest", "rf"), ("XGBoost", "xgb"), ("Rule", "threshold")]
EX = [(name, "exfil_" + d) for name, d in C2]
C2_DATA, EX_DATA = "data/dataset.npz", "data/exfil_dataset.npz"
EV = Path("results/evasion")


def load(d, data):
    return Detector.from_result(f"results/{d}", data)


def rate(k, n, ci=None):
    ci = ci or pm.wilson(k, n)
    return f"{k}/{n} [{ci[0]:.2f}, {ci[1]:.2f}]" if n else "-"


def ev_cell(row, time=False):
    k, n = row["evaded"]
    s = rate(k, n)
    if row["bracket"] and not time:
        s += f"  ({row['bracket'][0]:g}, {row['bracket'][1]:g}]  q={row['median_queries']:g}"
    if time and row["median_time"] is not None:
        s += f"  {row['median_time']:.1f} s"
    return s


def clean_table(title, dets, data, out):
    print(f"\n== {title}")
    print(f"{'detector':16s}{'acc':>7s}{'F1':>6s}{'recall':>8s}{'FPR':>8s}")
    for name, d in dets:
        m = pm.clean_metrics(load(d, data), data)
        out[d] = m
        print(f"{name:16s}{m['accuracy']:7.3f}{m['f1']:6.2f}{m['recall']:8.2f}{m['benign_fpr']:8.3f}")


def c2(out):
    clean_table("Table 5: clean C2 detection", C2, C2_DATA, out.setdefault("c2_clean", {}))
    sw = np.load(EV / "sweep.npz")
    _, held = split_instances(sw, 0.67)
    res = {d: pm.score(load(d, C2_DATA), sw) for _, d in C2}

    print("\n== Table 6: timing-jitter evasion, 30 instances (naive detected | evaded, bracket, queries)")
    t6 = out.setdefault("c2_jitter", {})
    for name, d in C2:
        row = t6[d] = pm.summary(res[d])
        above = int((res[d]["grid"][res[d]["first"][res[d]["evaded"]]] > 0.3).sum())
        row["n_above_0.3"] = above
        print(f"{name:16s}{row['naive_detected'][0]:>3d}/30  {ev_cell(row)}  "
              f"max={row['max_min']}  >0.3: {above}")
    tot = sum(t6[d]["evaded"][0] for _, d in C2), sum(t6[d]["evaded"][1] for _, d in C2)
    out["c2_total"] = tot

    p = out["c2_paired_matched_vqc"] = pm.paired(res["classical_matched"], res["quantum"])
    print(f"\n== Sec 4.3 paired matched MLP vs VQC: n={p['n']}  VQC harder {p['b_higher']}, "
          f"MLP harder {p['b_lower']}, tie {p['tie']}")
    tr = out.setdefault("c2_transfer", {})
    for src, dst in [("classical_native", "quantum"), ("quantum", "classical_native"),
                     ("classical_native", "rf"), ("classical_native", "xgb"), ("classical_native", "threshold")]:
        t = tr[f"{src}->{dst}"] = pm.transfer(res[src], res[dst])
        print(f"  transfer {src} -> {dst}: {rate(*t['transferred'], t['wilson'])}")

    print("\n== Table 7: single-axis hardening, held-out jitter evasion (undefended -> hardened)")
    t7 = out.setdefault("c2_hardened", {})
    for name, d in C2:
        h = load(d + "_hardened", C2_DATA)
        before = pm.summary(pm.score(load(d, C2_DATA), sw, held))
        after = pm.summary(pm.score(h, sw, held))
        cm = pm.clean_metrics(h, C2_DATA)
        t7[d] = {"clean": cm, "before": before, "after": after}
        print(f"{name:16s}acc {out['c2_clean'][d]['accuracy']:.3f}->{cm['accuracy']:.3f}  "
              f"FPR {out['c2_clean'][d]['benign_fpr']:.3f}->{cm['benign_fpr']:.3f}  "
              f"{rate(*before['evaded'])} -> {ev_cell(after)}")

    lines = {"timing": (sw, held), "size": (np.load(EV / "sweep_sizeaxis_full.npz"), held),
             "combined": (np.load(EV / "sweep_combined_heldout.npz"), None)}
    print("\n== Table 8: size-jitter escalation vs single-axis hardened (timing jitter 0.5, held-out)")
    t8 = out.setdefault("c2_sizeaxis", {})
    for name, d in C2:
        row = t8[d] = pm.summary(pm.score(load(d + "_hardened", C2_DATA), *lines["size"]))
        print(f"{name:16s}naive {row['naive_detected'][0]}/9  {ev_cell(row)}")

    print("\n== Joint timing+size hardening, held-out (single-axis -> joint)")
    tj = out.setdefault("c2_joint", {})
    for name, d in C2:
        j = load(d + "_joint_hardened", C2_DATA)
        cm = pm.clean_metrics(j, C2_DATA)
        tj[d] = {"clean": cm}
        cells = []
        for line, (s, idx) in lines.items():
            single = pm.summary(pm.score(load(d + "_hardened", C2_DATA), s, idx))
            joint = pm.summary(pm.score(j, s, idx))
            tj[d][line] = {"single": single, "joint": joint}
            cells.append(f"{line}: {rate(*single['evaded'])} -> {ev_cell(joint)}")
        print(f"{name:16s}acc {cm['accuracy']:.3f} FPR {cm['benign_fpr']:.3f}\n    " + "\n    ".join(cells))


def exfil(out):
    clean_table("Table 9: clean exfiltration detection", EX, EX_DATA, out.setdefault("exfil_clean", {}))
    sw = np.load(EV / "exfil_sweep.npz")
    ct = sw["completion_time"]
    _, held = split_instances(sw, 0.67)
    res = {d: pm.score(load(d, EX_DATA), sw) for _, d in EX}
    print(f"\n== Table 10: throttle evasion, 30 instances (naive completion median {np.median(ct[:, 0]):.2f} s)")
    t10 = out.setdefault("exfil_throttle", {})
    for name, d in EX:
        row = t10[d] = pm.summary(res[d], ct)
        print(f"{name:16s}{row['naive_detected'][0]:>3d}/30  {ev_cell(row)}  {row['median_time']:.1f} s")
    out["exfil_total"] = (sum(t10[d]["evaded"][0] for _, d in EX), sum(t10[d]["evaded"][1] for _, d in EX))
    p = out["exfil_paired_matched_vqc"] = pm.paired(res["exfil_classical_matched"], res["exfil_quantum"])
    print(f"\n  paired matched MLP vs VQC: n={p['n']}  VQC lower throttle {p['b_lower']}, "
          f"higher {p['b_higher']}, tie {p['tie']}")
    for src, dst in [("exfil_classical_native", "exfil_quantum"), ("exfil_quantum", "exfil_classical_native")]:
        t = out.setdefault("exfil_transfer", {})[f"{src}->{dst}"] = pm.transfer(res[src], res[dst])
        print(f"  transfer {src} -> {dst}: {rate(*t['transferred'], t['wilson'])}")

    print("\n== Table 11: throttle hardening, held-out (undefended -> hardened, time at evasion)")
    t11 = out.setdefault("exfil_hardened", {})
    for name, d in EX:
        h = load(d + "_hardened", EX_DATA)
        before = pm.summary(pm.score(load(d, EX_DATA), sw, held), ct[held])
        after = pm.summary(pm.score(h, sw, held), ct[held])
        cm = pm.clean_metrics(h, EX_DATA)
        t11[d] = {"clean": cm, "before": before, "after": after}
        print(f"{name:16s}acc {cm['accuracy']:.3f} FPR {cm['benign_fpr']:.3f}  "
              f"{rate(*before['evaded'])} -> {ev_cell(after, time=True)}")

    print("\n== Table 12: escalation sweeps vs hardened detectors (evaded, time at evasion)")
    t12 = out.setdefault("exfil_escalation", {})
    for tag in ("volume", "jitter", "chunk"):
        s = np.load(EV / f"exfil_esc_{tag}.npz")
        print(f"  ({tag})")
        for name, d in EX:
            row = t12.setdefault(d, {})[tag] = pm.summary(pm.score(load(d + "_hardened", EX_DATA), s),
                                                          s["completion_time"])
            print(f"    {name:16s}{ev_cell(row, time=True)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="also write every number to this JSON file")
    args = ap.parse_args()
    out = {}
    c2(out)
    exfil(out)
    print(f"\nTotal evaded/detected: C2 {out['c2_total'][0]}/{out['c2_total'][1]}, exfil "
          f"{out['exfil_total'][0]}/{out['exfil_total'][1]}, all "
          f"{out['c2_total'][0] + out['exfil_total'][0]}/{out['c2_total'][1] + out['exfil_total'][1]}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
