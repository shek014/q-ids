# Q-IDS: what to do after runs A–E

Run everything from the repo root. `.venv/bin/python` = your venv (or `source .venv/bin/activate`).

## 0. Back up the old results (once)
```
sudo chown -R $USER results capture data
cp -r results results_60runs        # old 60-run models + 9-instance sweeps, for reference
```
`data/dataset_60runs.npz` / `data/exfil_dataset_60runs.npz` are the old datasets.
Make sure `data/dataset.npz` and `data/exfil_dataset.npz` are the MERGED ones (~1260 / ~1440 flows):
```
.venv/bin/python -c "import numpy as np;[print(f,np.bincount(np.load(f)['y'])) for f in ['data/dataset.npz','data/exfil_dataset.npz']]"
```

## 1. Put the new sweeps under the default names
Every script (and paper/make_figures.py) reads the default names, so rename instead of passing paths:
```
mv results/evasion/sweep_v3.npz       results/evasion/sweep.npz
mv results/evasion/exfil_sweep_v3.npz results/evasion/exfil_sweep.npz
# results/evasion/sweep_sizeaxis.npz stays as is (run E)
```
Check none of them left a `*.partial.npz` behind (that means the run didn't finish).

## 2. Train every detector on the merged data
C2:
```
.venv/bin/python -m models.classical.train --out results/classical_native
.venv/bin/python -m models.classical.train --out results/classical_matched --pca-components 8
.venv/bin/python -m models.quantum.train   --out results/quantum
.venv/bin/python -m models.baselines.train --kind rf        --out results/rf
.venv/bin/python -m models.baselines.train --kind xgb       --out results/xgb
.venv/bin/python -m models.baselines.train --kind threshold --out results/threshold
```
Exfil:
```
D=data/exfil_dataset.npz
.venv/bin/python -m models.classical.train --data $D --out results/exfil_classical_native
.venv/bin/python -m models.classical.train --data $D --out results/exfil_classical_matched --pca-components 8
.venv/bin/python -m models.quantum.train   --data $D --out results/exfil_quantum
.venv/bin/python -m models.baselines.train --kind rf        --data $D --out results/exfil_rf
.venv/bin/python -m models.baselines.train --kind xgb       --data $D --out results/exfil_xgb
.venv/bin/python -m models.baselines.train --kind threshold --data $D --out results/exfil_threshold
```
Clean comparison table + plots: `.venv/bin/python -m evaluation.compare --out results/comparison`
Test sets are now ~190 (C2) / ~215 (exfil) flows. Take accuracy / FPR / recall for Table 1 and the exfil clean table from each `results/*/meta.json` and `test_predictions.npz`.

## 3. Evasion scoring (offline, no root)
MLP vs VQC (paired + transfer), C2 and exfil:
```
.venv/bin/python -m evasion.transfer
.venv/bin/python -m evasion.exfil_transfer
```
Baselines: the same scripts take any detector dirs, so score each baseline against the MLP
(the JSON keys "classical"/"quantum" then just mean "first"/"second" detector):
```
for b in rf xgb threshold; do
  .venv/bin/python -m evasion.transfer --classical results/classical_native --quantum results/$b \
     --out results/evasion/transfer_mlp_vs_$b.json
done
.venv/bin/python -m evasion.exfil_transfer --native results/exfil_rf --matched results/exfil_xgb \
   --quantum results/exfil_threshold --out results/evasion/exfil_transfer_baselines.json
# in that file: native_mlp = RF, matched_mlp = XGBoost, quantum = threshold
```

## 4. Adversarial training + held-out evaluation
```
.venv/bin/python -m evasion.harden
.venv/bin/python -m evasion.harden_eval --out results/evasion/harden_eval.json
.venv/bin/python -m evasion.exfil_harden
.venv/bin/python -m evasion.exfil_harden_eval --out results/evasion/exfil_harden_eval.json
```
`harden` / `exfil_harden` now also retrain RF / XGBoost / threshold into `results/*_hardened`.
`harden_eval` / `exfil_harden_eval` only list the MLP/VQC pairs — to include baselines, add
`("rf","rf_hardened"), ("xgb","xgb_hardened"), ("threshold","threshold_hardened")` to the list in
`harden_eval.py` (and the `exfil_`-prefixed ones in `exfil_harden_eval.py`).
Keep `--train-frac` at the default 0.67 everywhere — run E captured exactly the instances it holds out.

## 5. Size-axis escalation (run E) against every hardened detector
```
.venv/bin/python - <<'PY'
import json, numpy as np
from evasion.detector import Detector
from evasion.transfer import min_evading_jitter, _stats
s = dict(np.load("results/evasion/sweep_sizeaxis.npz")); s["jitters"] = s["values"]   # grid = size_jitter
out = {}
for d in ["classical_native_hardened","classical_matched_hardened","quantum_hardened",
          "rf_hardened","xgb_hardened","threshold_hardened"]:
    ev, mj, idx = min_evading_jitter(Detector.from_result(f"results/{d}"), s, 0.5)
    out[d] = {**_stats(ev, mj), "n": int(len(ev)),
              "median_queries": float(np.median(idx[ev] + 1)) if ev.any() else None}
    print(d, out[d])
json.dump(out, open("results/evasion/sizeaxis_all_hardened.json","w"), indent=2)
PY
```
"min jitter" in that output is the minimal evading size_jitter (timing jitter fixed at 0.5).

## 6. Figures
```
.venv/bin/python paper/make_figures.py
cp paper/figures/*.pdf paper/tex/figures/
```
Re-check every caption number afterwards (instance counts are now 30, not 9).

## 7. Analysis still NOT implemented (needed before submission)
- **Detected-only evasion:** count an instance as evaded only if the detector caught the unmodified
  attack (grid value 0). The VQC in particular misses many naive attacks, so its "100% evasion" is
  partly non-detection. Report the naive detection rate next to every evasion rate.
- **Equal-FPR comparison:** also compare detectors at the same benign false-positive rate
  (threshold set on the validation split), not only at P(benign) = 0.5.
- **Confidence intervals:** Wilson 95% intervals on every rate (evasion, transfer, FPR, recall).
- **Exfil deadline frontier:** `exfil_sweep.npz` now stores `completion_time` and
  `delivery_confirmed`, so for any deadline D ≤ 60 s a point is functional iff confirmed and
  completion_time ≤ D. Plot the fraction of instances that can still evade vs D, original vs
  hardened detectors. This decides whether exfil is a headline result or gets cut to an appendix.
- **Realism check:** run `features.extract` on a public capture with real NTP/keepalive traffic and
  compare `cv_iat` / throughput distributions to your synthetic benign confusers (one figure).

## 8. Decide the framing FROM the new numbers
- If RF/XGBoost resist evasion much better than the MLP → story becomes "what makes a detector robust".
- If the VQC's evasions are mostly non-detection → say that, not "no quantum advantage".
- If the deadline frontier bites → exfil is a headline contrast (C2 axes are free, exfil's have a budget).
- C2 threshold rule alone = ~70% acc / 40% FPR, exfil throughput rule alone = MLP-level → C2 detection
  is genuinely multi-feature yet beaten by one axis; exfil collapses to one rule. Worth stating.

## 9. Rewrite the paper (paper/tex/main.tex)
- Update every number/table in §4 (n = 30 instances, bigger test sets, add baselines + CIs).
- Small fixes: VQC exfil row ("0 (naive)" hides a config that needed 1.5); replace "perfect-on-paper";
  add the exfil subsection to the §4 intro sentence.
- Write Intro, Discussion (+Limitations), Conclusion, then Abstract (≤250 words), Highlights (3–5,
  ≤85 chars each). The paper/*.md drafts use OLD numbers — don't copy their claims.
- Fill the `\todo{cite C2 framework jitter documentation}` (Cobalt Strike sleep/jitter, Sliver docs).
- references.bib: no `% comment` on the same line as an `@entry{key,` (breaks BibTeX).

## 10. Submission admin
- Author/affiliation: SRM Institute of Science and Technology, Vadapalani, Chennai, India; decide email.
- Co-author/supervisor + CRediT roles.
- Archive code + data on Zenodo → DOI into Methodology and Data Availability.
- Clean elsarticle build (no `??`, no `[TODO`), cover letter, Highlights in COSE's separate field.

## Exfil rerun (testbed fix, 26 Oct)
The first exfil sweep ran queries in a 2-host network with no background traffic, so its naive exfil
was faster and more regular than any training exfil (out of distribution; matched MLP + XGBoost called
20/30 naive exfils benign despite ~0.95 test recall). `evasion/exfil_capture_helper.py` now defaults to
`--topology full`: the query runs inside the training scenario's network with its background traffic.
1. Sanity check one naive query (compare bytes_per_second / std_iat with training exfil: median ~360K, ~0.007):
   `sudo .venv/bin/python -m evasion.exfil_query --volume 1150000 --throttle 0 --seed 3`
2. Rerun D (old sweep kept as results/evasion/exfil_sweep_minimal.npz):
   `mv results/evasion/exfil_sweep.npz results/evasion/exfil_sweep_minimal.npz`
   `systemd-inhibit sudo .venv/bin/python -m evasion.exfil_sweep --volumes 800000,1150000,1500000 --seeds-per-config 10 --throttles 0,0.5,1,1.5,2,2.5,3,5,8,12 --out results/evasion/exfil_sweep.npz`
3. Redo step 3 (exfil_transfer + baselines) and step 4 (exfil_harden / exfil_harden_eval) for exfil.
C2 queries have the same 2-host setup, but the shift is in the "more detectable" direction (rigid sweep
beacons are more regular than training ones). Either mention it as a limitation, or port the same
`--topology full` mode to evasion/capture_helper.py and rerun C + E.

## Git hygiene
- Add `.claude/` and `*.patch` to `.gitignore`.
- Commits are authored by your own git config; no co-author lines needed.
