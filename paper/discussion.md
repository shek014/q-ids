# 5. Discussion

> Draft — Discussion + Limitations (§5), written for the paper (reviewers / co-author). Core argument
> is C2-only (the committed study); paragraphs marked **[exfil]** depend on including the data-
> exfiltration second attack family (§X / paper/exfil_section.md) and trim cleanly if the co-author
> scopes it out. `[cite]`/`[TODO]` as elsewhere.

## 5.1 What the results mean for flow-level NIDS

Our central finding is that a flow-level machine-learning intrusion detector, even at a *realistic*
operating point (≈86% accuracy, not a saturated 100%), is defeated for 100% of instances by a
perturbation that is both **realizable** — carried out as actual traffic, not a feature-vector edit —
and **functionality-preserving**. The attacker never touches the detector, the feature pipeline, or
other hosts; they change one parameter of their own traffic that the protocol leaves free, and a
functional oracle confirms the attack still works throughout.

The deeper point is *why* this is possible, and it is not a defect of a particular model or feature
set. Flow-level detection of an evasion-amenable attack rests on a statistic of the attacker's own
traffic — here the timing regularity of a C2 beacon (`cv_iat`). That statistic is exactly a knob the
attacker controls and can move at negligible functional cost. A detector that keys on it is therefore
keying on something the adversary can simply dial away. The realistic benign traffic makes this
concrete: because regular benign services (NTP, health checks, keepalives) populate the same region a
jittered beacon moves into, the detector cannot tighten its boundary to follow the beacon without
flagging benign traffic — the 13–18% false-positive rate is the price of catching rigid beacons, and
it *bounds* how far any such detector can chase the attack. Evasion is not just possible; it is cheap
and it lands the attacker in a region the defender cannot safely police.

## 5.2 The quantum classifier offers no robustness advantage

A feature-space literature has raised the possibility that quantum classifiers are intrinsically more
robust, learning representations that classical adversarial perturbations do not disturb
[cite:west2023nature]. Our results do not support this in the problem space. On the realistic task the
VQC is, if anything, *easier* to evade than the classical MLP, and we observed an apparent quantum
"edge" only on an earlier saturated (100%-clean-accuracy) version of the task — an artifact of trivial
separability that vanished once detection was made realistic. We read this as a caution about
transferring feature-space robustness claims to operational settings: a classifier whose decision
geometry resists $L_p$ perturbations of the *encoded* input need not resist a *realizable* change to
the traffic, because the realizable change is not an $L_p$ ball around a fixed point — it is a move
along an attacker-controlled axis to a genuinely benign-populated region. Robustness measured in the
space the attacker cannot actually reach does not predict robustness in the space they can.

## 5.3 The limits of adversarial training

Adversarial training is the natural defense, and it is partially effective: folding evaded instances
back into training drives the evaded axis's success rate to zero for the classical MLP. But it does
not restore robustness in any durable sense, for two reasons our results make explicit. First, it is
not free: teaching the detector that a perturbed attack is malicious pushes it to flag the
near-identical benign traffic, raising the false-positive rate (most sharply for the VQC, whose benign
FPR we saw climb substantially). **Robustness and defensibility are not the same**: a model can be made
to resist one evasion only by becoming less usable. Second, for C2 the defense merely *relocates* the
vulnerability — hardening the timing axis leaves the payload-size axis free, and the hardened detector
is re-broken for 100% of instances by size padding alone. Realizable traffic offers several
functionally-free axes, and single-axis adversarial training defends them one at a time while the
attacker simply steps to the next. A defense that actually closed the gap would have to cover the joint
space of realizable perturbations, not a single axis — a substantially harder problem, and one a
false-positive budget may not permit.

## 5.4 Generality across attack families **[exfil]**

The same picture holds for a second, structurally different attack. Data exfiltration shares none of
C2's mechanics: its detectable signal is *throughput* rather than timing regularity, its evasion axis
is rate throttling rather than jitter, and — unlike jitter — throttling carries a real functional cost,
because the payload must still arrive before a deadline. Yet the conclusions are unchanged: a realistic
detector is evaded for 100% of instances, the VQC again shows no robustness benefit (indeed it is
strictly easier to evade, often failing to detect even the naive attack), transfer is again asymmetric
and follows detector strength, and adversarial training again defeats the single axis for the classical
model at a false-positive cost while leaving the VQC both evadable and more false-positive-prone. That
two attacks with different signals, different evasion axes, and different evasion *costs* reach the same
conclusion suggests the vulnerability is a property of flow-level learned detection under problem-space
constraints — not an artifact of one attack.

## 5.5 Limitations

We state the threats to validity plainly.

- **Emulated testbed.** Traffic is generated and captured in Mininet, not on a production network.
  This buys the controllability problem-space evasion requires — one cannot manipulate a static
  benchmark in the loop — but the absolute accuracy and false-positive numbers are properties of our
  benign mix, not of any real deployment. The *relative* findings (evasion succeeds; no quantum edge;
  defense relocates) are the contribution; the absolute rates are illustrative. External validity on a
  public benchmark (e.g. CIC-IDS/UNSW) is left to future work [cite].
- **Simulated quantum classifier.** The VQC is trained and evaluated on a noiseless state-vector
  simulator with 8 qubits and a shallow ansatz. Real NISQ hardware, noise, and larger circuits could
  change its clean accuracy, though our robustness conclusion (no problem-space advantage) is about the
  decision geometry a well-trained VQC induces, not about hardware constraints [cite].
- **Window-dependent features.** Several features (`duration`, `packet_count`, `byte_count`) partly
  reflect the fixed capture window rather than intrinsic flow structure; our queries use the same 60 s
  window as training to stay in-distribution. A window-invariant feature set would strengthen external
  validity and enable cheaper queries, and is a natural extension [TODO: "B"].
- **Model and threat scope.** We study a from-scratch MLP and a VQC, not the full range of deployed
  NIDS architectures (deep nets, ensembles, sequence models). The attacker is black-box at the verdict
  level and does not reduce the attack's essential payload (a beacon still checks in; an exfil still
  moves its full volume).
- **[exfil] Untested escalation.** For exfil we demonstrate that adversarial training defeats the
  throttle axis for the classical MLP, but we have not yet run the second-axis escalation that, for C2,
  showed the defense merely relocates. Whether exfil tells the identical "relocation" story is
  therefore still open (§X.4).

## 5.6 Future work

Beyond the items above — real-traffic / benchmark validation, window-invariant features, the exfil
axis escalation, and NISQ-hardware evaluation — the most consequential open problem is **joint-space
defense**: whether a detector can be made robust across *all* functionally-free realizable axes at once
without an unacceptable false-positive cost. Our results suggest single-axis hardening cannot, and that
the defender's false-positive budget, not the attacker's effort, is the binding constraint.
