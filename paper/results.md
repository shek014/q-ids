# 4. Results

> Draft — Results section. Written for the paper (reviewers / co-author). All numbers are from the
> realistic-benign dataset (420 flows; 70/15/15 stratified split) and 15 beacon instances per line
> search. Figures/tables referenced as [Table N] / [Fig. N] are to be rendered from the committed
> results. `[TODO]` marks a check to confirm with the co-author.

We report, in order: clean detection performance (4.1); problem-space evasion via timing jitter
(4.2); the paired classical-vs-quantum comparison and transferability (4.3); and the
adversarial-training defense with its axis escalation (4.4).

## 4.1 Clean detection performance

On unmodified traffic the detectors are accurate but *not* saturated, as intended by the realistic
benign mix (Section 3.2). On the held-out test set [Table 1]:

| Detector | Input | Accuracy | C2 F1 | C2 recall | Benign FPR |
|---|---|---|---|---|---|
| MLP (native) | 17 features | 0.857 | 0.78 | 0.89 | 0.156 |
| MLP (matched) | PCA-8 | 0.873 | 0.80 | 0.89 | 0.133 |
| VQC | PCA-8 | 0.746 | 0.56 | 0.56 | 0.178 |

The classical MLP reaches ≈86–87% accuracy; the quantum VQC is markedly weaker at 74.6%, and its C2
F1 of 0.56 shows it detects only about half of the beacons. The non-trivial benign false-positive
rates (13–18%) are the honest cost of the task: the rigid benign traffic (NTP/health-check
analogues) is nearly indistinguishable from a beacon at the flow level, so a detector that catches
rigid beacons necessarily mislabels some rigid benign flows. This is the realistic operating point a
flow-level C2 detector faces, and it is the regime in which we evaluate evasion.

## 4.2 Problem-space evasion via timing jitter

We line-search the timing-jitter axis (interval and size held fixed; Section 3.5) against each
detector over 15 beacon instances, recording the minimal jitter at which the beacon is classified
benign while remaining functional. The functional oracle confirmed that **every** evading
configuration remained operational (≥ 2 successful check-ins, no gap > 30 s): evasion is
functionality-preserving throughout.

Both detectors are **evaded for 100% of beacon instances** at a trivial, cost-free level of timing
jitter (median minimal jitter 0.20 — i.e. ≈ ±20% randomization of an otherwise-regular check-in
interval) [Table 2, Fig. 1]. Figure 1 also shows that the VQC assigns P(benign) ≈ 0.44 to even a
*rigid* beacon — it barely detects them at all, consistent with its 0.56 C2 recall — whereas the MLP
starts confident (≈ 0.12) but crosses the decision threshold by jitter ≈ 0.2. Randomizing check-in timing by this amount is a change real C2 malware
already performs to appear less robotic, and the beacon continues to reach its C2 throughout. A
perfect-on-paper detection pipeline is therefore defeated, every time, by a change that costs the
attacker essentially nothing.

## 4.3 Classical vs. quantum: paired comparison and transferability

Because the quantum and classical detectors are trained on identical inputs (Section 3.4) and scored
on the *same* captured flows via the capture-once sweep (Section 3.6), their robustness is directly
comparable.

**No quantum robustness advantage.** Across the 15 paired beacon instances, the minimal evading
jitter was **equal for the two detectors in 10 cases and lower for the VQC in 5, and never higher**
(mean difference VQC − MLP = −0.037). On a realistic task the quantum classifier is thus, if
anything, *marginally easier* to evade, and is in no case harder. This reverses an apparent
quantum "edge" we observed on an earlier saturated (100%-clean-accuracy) version of the task, which
we attribute to that task being trivially separable; once detection is realistic, the apparent
advantage disappears.

**Transferability is asymmetric and follows detector strength** [Table 3]. A beacon tuned to just
evade the VQC also evaded the MLP in only 67% of cases, whereas a beacon tuned to just evade the MLP
evaded the VQC in 100% of cases. That is, evasions crafted against the (weaker, lower-jitter) VQC do
not reliably carry to the stronger MLP, while evasions against the MLP transfer fully to the VQC —
consistent with the VQC's decision region being the more permissive of the two. An attacker who
succeeds against the stronger detector succeeds against the weaker one for free.

## 4.4 Adversarial-training defense and axis escalation

We fold evaded beacons into training (beacon seeds 0–9) and retrain, holding out seeds 10–14 for
evaluation (Section 3.7).

**Hardening defeats the timing-jitter attack — at a cost.** On the held-out instances, the
jitter evasion that previously succeeded for 100% of beacons succeeds for **0%** against both
hardened detectors. However, hardening raises the benign false-positive rate and, for the VQC,
degrades clean accuracy [Table 4]:

| Hardened detector | Clean accuracy | Benign FPR | Held-out jitter evasion |
|---|---|---|---|
| MLP (native) | 0.825 | 0.178 | 100% → 0% |
| MLP (matched) | 0.857 | 0.156 | — |
| VQC | 0.714 | 0.311 | 100% → 0% |

Teaching the detector that jittered beacons are malicious pushes it to flag the near-identical rigid
benign traffic: the VQC's benign FPR rises to 31%. Defending the timing axis is not free.

**The defense is circumvented by escalation to a second axis.** The jitter hardening can only exploit
the *timing* axis, so we hold timing jitter at an already-evading value and line-search the
previously-unused **payload-size-jitter** axis against the hardened MLP (Section 3.7). The hardened
MLP — which the timing attack could no longer touch (0% evasion) — is **re-broken for 100% of beacon
instances** at a trivial amount of size padding (median minimal `size_jitter` = 0.30, median 4
queries to evasion) [Table 5, Fig. 2]. Size padding is as functionally free as timing jitter: the beacon's
control channel is unaffected.

Single-axis adversarial training therefore does not close the vulnerability; it **relocates** it to
the next functionally-free axis. Because realizable traffic offers the attacker several such axes
(timing and size here, among others), defending them one at a time is a losing exchange — the
attacker simply moves to an undefended axis at negligible cost.
