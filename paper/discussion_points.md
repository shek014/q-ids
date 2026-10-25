# Discussion — writing points (§5)

> Working guide for writing §5 in your own words. All numbers are final (post-rerun). This is scratch —
> delete it (and the other draft `.md` files) before the v1.0 Zenodo release so it doesn't ship.
> Supersedes the framing in the older `discussion.md`, whose hardening story is now out of date.

## The spine (what the whole section argues)
A realistic flow-level ML-NIDS (~75–87% clean) is fully evaded (100%) by a realizable,
functionality-preserving change; the quantum model gives no robustness benefit; and single-axis
adversarial training does not durably close the hole. The binding constraint for the defender is its
own false-positive budget and the attacker's supply of free axes — not the attacker's effort.

## 5.1 What it means for flow-level NIDS
- The detector works at a believable level on normal traffic yet collapses completely under evasion —
  that gap is the finding, not a contradiction.
- *Why* it's possible: flow-level detection of an evasion-amenable attack rests on a statistic of the
  attacker's own traffic (`cv_iat` for C2, `bytes_per_second` for exfil) that the attacker controls with
  functional slack. Keying on it means keying on something the adversary can dial away.
- The realistic benign confuser makes it concrete: a jittered beacon / throttled exfil lands among
  genuinely benign traffic, so the detector can't tighten without flagging benign flows — the FPR is the
  price of catching the rigid/loud version, and it *bounds* how far any such detector can chase.
- Numbers: clean acc C2 0.857 (MLP) / 0.746 (VQC), exfil 0.847 / 0.833; evaded 100% in every case.

## 5.2 No quantum robustness advantage
- VQC is equal-or-easier to evade in both attacks (paired: C2 MLP harder 7/9, mean −0.11; exfil
  matched-MLP harder 6/9, mean −1.08). Never meaningfully harder.
- The apparent "edge" on the earlier saturated (100%-clean) task was an artifact of trivial
  separability; it vanished once detection was realistic.
- The caution to draw: feature-space robustness claims (e.g. West et al., L_p on the encoded input)
  do not transfer to a *realizable* attack, which is a move along an attacker-controlled axis into a
  benign-populated region, not an L_p ball around a fixed point.

## 5.3 The limits of adversarial training  ← THIS IS WHAT CHANGED; use these numbers
- **C2: partial, then relocates.** Hardening reduces jitter evasion 100% → 33% (one of three held-out;
  not 0%) and raises the required jitter, at a real cost (both MLPs lose accuracy; FPR 20–29%). Then
  holding jitter at an evading value and attacking the size axis re-breaks the hardened MLP 100% (5/5)
  at size-jitter 0.15. So it neither fully closes the defended axis nor prevents relocation.
- **Exfil: a different failure mode.** Hardening has *no clean cost* (accuracy even improves: native
  0.847→0.875) — a throttled exfil stays more distinct from benign than a jittered beacon does. But it
  mostly *fails*: the deployed native MLP and the VQC stay 100% evadable (the attacker just throttles
  harder, 2.0→5.0, still within deadline); only the matched (PCA-8) MLP is defended (0%). No second
  axis is needed — the throttle axis has headroom up to the deadline.
- **Unify it:** single-axis hardening doesn't durably work — C2 by relocation, exfil by same-axis
  headroom. "Robustness ≠ defensibility." A durable defense would need the *joint* space of free axes
  without exhausting the FP budget.
- Caveat to state here: held-out set is 3 instances, so these rates are coarse/directional.

## 5.4 Generality across attack families
- Two structurally different attacks — different signal (timing vs throughput), different axis (jitter
  vs throttle), different cost (free vs deadline-bounded) — reach the same core conclusion.
- The *contrasts* (exfil's genuine functional cost; its different hardening behaviour) strengthen the
  claim: the vulnerability is a property of flow-level learned detection under problem-space
  constraints, not an artifact of one attack.

## 5.5 Limitations (own them plainly — don't let a reviewer raise them first)
- **Emulated testbed (the big one):** absolute accuracy/FPR are properties of our Mininet benign mix,
  not a real network. The *relative* findings (evasion succeeds; no quantum edge; hardening doesn't
  close it) are the contribution; the absolute rates are illustrative. Benchmark validation is future
  work (→ the planned CIC-IDS/UNSW companion).
- **Small samples:** evasion n=9 (3 configs × 3 seeds), hardening held-out n=3, test sets 63 (C2) /
  72 (exfil) flows → report as directional, no CIs.
- **Simulated VQC:** noiseless 8-qubit state-vector simulator, shallow ansatz.
- **Window-dependent features** (duration/packet_count/byte_count) at a fixed 60 s window.
- **Scope:** one from-scratch MLP + one VQC, not the full range of deployed NIDS; exfil second-axis
  escalation not run.

## 5.6 Future work
- The real open problem: **joint multi-axis defense** without an unacceptable FP cost.
- Benchmark cross-check (CIC-IDS2017 / UNSW-NB15) for external validity — the companion paper.
- NISQ-hardware VQC; window-invariant features; exfil axis escalation (chunk-size / multi-connection).

## One-sentence takeaway (reuse in Conclusion/Abstract)
Flow-level learned detection of evasion-amenable attacks is fragile under realizable,
functionality-preserving evasion; a quantum classifier offers no remedy; and single-axis adversarial
training does not close the gap — the defender's false-positive budget, not the attacker's effort, is
the binding constraint.
