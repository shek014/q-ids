# 1. Introduction

> Draft — Introduction (§1), written for the paper (reviewers / co-author). This is the most
> framing-sensitive section: it leads with **problem-space adversarial robustness of ML-NIDS** and
> treats the quantum classifier as *one of the compared models*, not as the headline — the emphasis is
> a decision to confirm with the co-author. Paragraphs/contributions marked **[exfil]** depend on
> including the data-exfiltration second attack family and trim cleanly if scoped out. `[cite]`/`[TODO]`
> as elsewhere.

Machine-learning classifiers are now a standard component of network intrusion detection systems
(NIDS), where they flag malicious traffic from flow-level statistics with high reported accuracy
[cite:nids-survey2023]. As with ML elsewhere, this raises the question of adversarial robustness: can
an attacker who understands the detector evade it? A large literature answers "yes" — but largely in
the **feature space**, by perturbing the flow's feature vector directly until the classifier flips
[cite:nids-survey2023; cite:ennaji2024]. Such demonstrations are informative about a model's decision
geometry, yet they sidestep the question a defender actually faces, because a perturbed feature vector
need not correspond to any traffic an attacker could send: the perturbation may violate protocol
constraints, break the attack's function, or simply have no realizable pre-image on the wire. Indeed,
recent work argues that many reported NIDS evasions do not survive real-world constraints and
**remain impractical** in deployment [cite:apruzzese2022; cite:elshehaby2023].

The sharper question is therefore one of **problem-space** robustness [cite:pierazzi2020]: robustness
to perturbations that are physically *realizable* as traffic and that *preserve the attack's
function*. Problem-space adversarial ML has been developed primarily for malware and other engineered-
feature domains [cite:pierazzi2020; cite:fence2022]; for network traffic it is only beginning to be
studied [cite:catillo2024], and typically still works from static datasets rather than manipulating
traffic in a live loop. This matters because the realizability constraint is not a mere technicality —
it determines which attacks are evadable at all. Volumetric or structural attacks (floods, scans)
cannot be hidden per flow because their detectable signature *is* their function. The interesting case
is an attack whose function is *separable* from its traffic shape, so the shape can be changed while
the function survives.

Command-and-control (C2) beaconing is exactly such an attack. Its function — maintaining a control
channel by periodic check-ins — is independent of the *regularity* of those check-ins, which is the
very thing a flow-level detector keys on (a low coefficient of variation of inter-arrival times)
[cite:yilmaz2026]. An attacker can randomize the timing to look less robotic without impairing the
beacon, walking it out of the detector's reach. We make this contested boundary realistic rather than
a straw man by populating the benign traffic with the regular services (NTP, health checks,
keepalives) that genuinely resemble a beacon, so that clean detection is accurate but not saturated —
and so that an evaded beacon lands among traffic the defender cannot safely flag.

Orthogonally, a growing body of work applies **quantum** and quantum-classical models — variational
quantum classifiers (VQCs) in particular — to intrusion detection, generally reporting competitive
clean accuracy [cite:abreu2024; cite:thomos2026]. Separately, a quantum-ML-theory literature has
suggested that quantum classifiers may be *intrinsically more adversarially robust* than classical
ones, learning features that classical attacks do not disturb [cite:west2023nature]. But that evidence
is entirely feature-space, and the QML-IDS literature evaluates clean accuracy, not robustness. Whether
a quantum detector offers any robustness advantage against a *realizable* attack — the thing an
operator cares about — has not, to our knowledge, been tested.

This paper addresses both gaps with a single study. We build a **model-in-the-loop, functionality-
preserving evasion** that operates entirely in the problem space: each candidate perturbation is
realized as actual traffic in an emulated network, its flow is extracted by the same pipeline the
detector was trained on, and it is admitted only if a **functional oracle** confirms the attack still
works. On top of this we compare a classical multilayer perceptron and a quantum-classical VQC trained
on **matched inputs**, so that any robustness difference is attributable to the model rather than the
information it receives. We evaluate minimal realizable evading perturbation, queries-to-evasion,
cross-model transferability, and the effect of adversarial-training defenses — on C2 beaconing, and
**[exfil]** on a second, structurally different attack family (data exfiltration, evaded by throughput
throttling rather than timing jitter) to test whether the findings generalize.

Our results are consistent and, for a defender, unwelcome. At a realistic operating point, both
detectors are evaded for **100%** of instances by a trivial, functionally-free perturbation. The
quantum classifier provides **no robustness advantage** — an apparent edge on a saturated
(100%-clean-accuracy) version of the task disappears once detection is made realistic, and on the
harder task the VQC is if anything easier to evade. Adversarial training does not restore robustness:
it drives one axis's evasion to zero at a measurable false-positive cost, but the vulnerability
**relocates** to the next functionally-free axis, which re-breaks the hardened detector completely.

We make the following contributions:

- **A problem-space, functionality-preserving evasion for flow-level NIDS**, driven model-in-the-loop
  against real traffic with a functional oracle — not a feature-space perturbation — together with a
  realistic benign mix that keeps the contested boundary genuine.
- **The first problem-space robustness comparison of a quantum vs. classical detector on a security
  task**, under a matched-input protocol, finding no quantum robustness advantage and showing that an
  apparent advantage on a saturated task is an artifact of trivial separability.
- **An analysis of adversarial-training defenses** showing that single-axis hardening relocates rather
  than closes the vulnerability, at a false-positive cost — i.e. that robustness and defensibility
  differ, and that a durable defense would need the joint space of realizable axes.
- **[exfil] Evidence of generality** across two attack families with different detectable signals,
  different evasion axes, and different evasion costs, reaching the same conclusions — indicating the
  vulnerability is a property of flow-level learned detection under problem-space constraints, not of a
  single attack.

All code and captured data are released for reproducibility [cite: repo/Zenodo DOI]. The rest of the
paper is organized as follows: §2 reviews related work; §3 details the testbed, detectors, and
problem-space evasion method; §4 reports results; §5 discusses implications and limitations; and §6
concludes.
