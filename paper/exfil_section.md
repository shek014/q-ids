# Second attack family: data exfiltration (DRAFT)

> Draft — the data-exfiltration second-attack family, written for the paper (reviewers / co-author).
> Self-contained so it can be slotted into the main paper **if the co-author approves the scope**:
> the methodology pieces extend §3 (threat model / testbed / evasion), and §X is a new results
> subsection paralleling §4. All numbers are from the realistic-benign exfil dataset (480 flows,
> 70/15/15 stratified) and the 15-seed capture-once throttle sweep. `[cite]`/`[TODO]` as elsewhere.

The purpose of this family is **generality**: to test whether the problem-space evasion result is
specific to C2 beaconing or holds for a structurally different attack with a different evadable signal
and — crucially — a different *cost* of evasion. Data exfiltration is a strong contrast case: where a
C2 beacon's detectable signal is *timing regularity* (evaded essentially for free by adding jitter),
an exfiltration flow's detectable signal is *throughput*, and the evasion (throttling) trades directly
against a functional deadline. The entire C2 testbed, feature pipeline, detectors, and matched-input
protocol (§3.1–§3.4) are reused unchanged; only the attack, its benign confuser, and the evasion axis
differ, as described below.

## Methodology additions

### Threat model (extends §3.1)
The attacker has staged a body of data on a compromised internal host and must move it to an external
collector. A flow-level NIDS classifies each flow as benign or exfiltration. The attacker's goal is
again to evade detection while **preserving functionality** — here, the *entire payload must reach the
collector within an operational deadline*. The attacker **can** manipulate the realizable transfer
parameters of their own traffic — the transfer rate, the chunking, and the inter-chunk timing — but
**cannot** reduce the total volume that must leave (the payload is fixed), nor alter other hosts'
traffic or the detector. A perturbation is admissible only if the full payload is delivered within the
deadline (the functional oracle below).

### Testbed and traffic (extends §3.2)
A single additional generator produces exfiltration as one TCP transfer of a fixed payload
`volume` to a collector, paced to a target `rate` (bytes/s) by inter-chunk delays. A **naive** exfil
dumps the payload as fast as it can: a short, high-throughput flow. Benign traffic is enriched, as for
C2, to make detection realistic rather than saturated. The decisive confuser is **high-rate benign
bulk** — legitimate backups, large downloads, and database replication — drawn from the **same volume
band and the same high-rate band** as the naive exfil. Because benign bulk and naive exfil share
volume *and* overlap in throughput, a flow-level detector cannot separate them cleanly: catching naive
exfil necessarily means false-flagging some benign bulk. (The benign mix also spans low-rate trickle —
telemetry/log-shipping — and moderate transfers, plus the non-transfer background of §3.2.) As with
C2, the only systematic separator between the attack and its nearest benign class is the axis the
evasion will manipulate: here, throughput.

The flow features and labelling are exactly those of §3.3 (the transfer is one flow keyed by source
MAC). The dataset comprises 480 flows (360 benign, 120 exfil) split 70/15/15, stratified.

### Problem-space evasion and its functional oracle (extends §3.5)
The evasion axis is **throttle**: the attacker lowers the transfer rate to push the flow's
`bytes_per_second` down out of the high-rate band and into the benign background. We parameterize it
as an effective rate `= base_rate / (1 + throttle)`, so `throttle = 0` is the naive dump and larger
values are slower and stealthier. The line search (§3.5) sweeps throttle upward for the minimal
evading value, exactly as for jitter.

The key structural difference from C2 is the **functional oracle**. For a beacon, raising jitter is
essentially free. For exfiltration it is not: throttling stretches the transfer, and the full payload
must still arrive before the deadline. The exfil is deemed **functional** iff the cumulative bytes
delivered to the collector reach `volume` within the deadline (`MAX_DELIVERY` = 60 s here). Throttle
too far and the transfer cannot finish in time — an inadmissible perturbation. The evasion must
therefore find a rate that is low enough to evade yet high enough to complete: the functional
constraint can, in principle, *defeat* the attack, which it cannot for C2.

## §X Results: data exfiltration

### X.1 Clean detection
On the realistic benign mix the detectors are accurate but not saturated [Table X1]. The classical MLP
reaches 84.7% with high exfil recall (0.94) at the cost of an 18.5% benign false-positive rate — it
catches almost every naive exfil, but false-flags some of the high-rate benign bulk it cannot
distinguish from it. The VQC reaches 83.3% but with markedly lower exfil recall (0.67): it misses
about a third of naive exfils outright.

| Detector | Input | Accuracy | Exfil recall | Exfil F1 | Benign FPR |
|---|---|---|---|---|---|
| MLP (native) | 17 features | 0.847 | 0.94 | 0.76 | 0.185 |
| MLP (matched) | PCA-8 | 0.847 | 0.94 | 0.76 | 0.185 |
| VQC | PCA-8 | 0.833 | 0.67 | 0.67 | 0.111 |

### X.2 Throttle evasion
Against every detector, exfiltration is **evaded for 100% of instances** by throttling [Table X2]. For
the classical MLP the median minimal throttle is 2.0 — the attacker slows from the naive 600 KB/s to
200 KB/s, a 3× slowdown — and the transfer still completes in ~6 s, far inside the 60 s deadline, so
the evasion is **functionality-preserving**: here the functional constraint does not bite at the
evading point. The naive exfil is confidently detected (P(benign) ≈ 0.12), and a modest,
functionally-free reduction in rate walks it across the boundary into the benign-bulk/trickle region.

| Detector | Evasion rate | Median min. throttle | Effective rate at evasion |
|---|---|---|---|
| MLP (native) | 100% | 2.0 | 200 KB/s |
| MLP (matched) | 100% | 2.0 | 200 KB/s |
| VQC | 100% | 0.0 | 600 KB/s (naive) |

### X.3 No quantum advantage, and transfer
The paired comparison on matched inputs is stark: across all 15 instances the **VQC required *less*
throttling than the matched MLP** (mean difference VQC − MLP = −2.0; VQC lower in 15/15, never higher).
The VQC's median minimal throttle is **0.0** — for a majority of instances the *naive* exfil already
evades it, a direct consequence of its weak clean recall (0.67). As in the C2 study, there is **no
quantum robustness advantage**; if anything it is sharper here, with the VQC strictly easier to evade
on every instance.

Transferability again follows detector strength [Table X3]: an evasion tuned to just beat the
(stronger) MLP evades the VQC in **100%** of cases, whereas an evasion tuned to just beat the (weaker)
VQC — often the naive exfil itself — evades the MLP in **0%** of cases. An attacker who defeats the
stronger detector defeats the weaker one for free; the converse never holds.

### X.4 Takeaway
Across a second, structurally different attack family — different detectable signal (throughput vs.
timing), different evasion axis (throttle vs. jitter), and a genuine functional *cost* to the evasion
that C2 lacked — the conclusions are unchanged: a realistic flow-level detector is fully evaded by a
realizable, functionality-preserving perturbation, and the quantum classifier provides no robustness
benefit (indeed less). The vulnerability is a property of flow-level learned detection under
problem-space constraints, not of any one attack.

> [TODO, optional] Hardening round: adversarially train on throttled exfil and report the FPR cost;
> a clean *second* realizable axis for the escalation (as C2's jitter→size) is less obvious for exfil
> (candidates: chunk-size/padding, multi-connection splitting) — decide with co-author whether to
> include.
> [TODO] Fold exfil into the figures (a second evasion curve, or a combined two-attack panel).
