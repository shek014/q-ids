# 2. Related Work

> Draft — Related Work section (pre-staged for the co-author to refine; the narrative emphasis is
> hers to shape). Written for the paper (reviewers / co-author). `[cite:key]` markers point at the
> reference list at the foot of this file; `[TODO]` marks a decision or a metadata check to confirm.
> Bibliographic details marked ✓ were verified against the publisher/DBLP; those marked ⚠ still need
> a final check.

Our work sits at the intersection of four strands: problem-space (realizable) adversarial machine
learning, adversarial evasion against network intrusion detection (NIDS), machine-learning detection
of C2 beaconing, and quantum machine learning for security together with its (separately studied)
adversarial robustness. We review each and position our contribution against it.

## 2.1 Problem-space vs. feature-space adversarial attacks

Most adversarial-ML results perturb the *feature vector* directly and measure success in feature
space. Pierazzi et al. [cite:pierazzi2020] formalized the alternative **problem-space** setting, in
which the attacker must produce a real object (there, an Android app) and the perturbation is
constrained by what is physically realizable — available transformations, preserved semantics,
absent artifacts, and plausibility — because in most domains there is no clean inverse map from a
desired feature vector back to a valid object. Chernikova and Oprea's FENCE [cite:fence2022] pushed
in the same direction for deep models in *constrained* domains, generating evasions that respect
feature inter-dependencies rather than treating features as freely perturbable, with network
security among the case studies.

These works establish the discipline we adopt — realizability and functionality preservation as
first-class constraints — but they were developed for malware and engineered-feature settings. We
carry the problem-space discipline into **network flow** NIDS with an *in-the-loop traffic
generator*: every candidate perturbation is realized as actual packets in an emulated network and
admitted only if a functional oracle confirms the attack still works (Section 3.5). We never perturb
a feature vector; we perturb the traffic and let the feature extractor do what it would in
deployment.

## 2.2 Adversarial evasion against NIDS

Adversarial attacks on ML-based NIDS are by now a large literature, surveyed in
[cite:nids-survey2023; cite:nids-challenges2024; cite:nids-understanding2026]. A recurring critique
is that a large share of reported successes are demonstrated in feature space or under idealized
threat models that an attacker could not actually realize on the wire. Apruzzese et al.
[cite:apruzzese2022] argued for realistic threat models with limited attacker knowledge and control,
and elShehaby and Matrawy [cite:elshehaby2023] went further, contending that evasion attacks
**remain largely impractical** against real ML-NIDS, especially dynamic ones. A 2024 line of work
begins to close this gap with explicitly problem-space NIDS attacks [cite:realistic-ps2024], and
related efforts restrict perturbations to attacker-controllable fields: the controllability-aware
framework of [cite:controllability2026] partitions flow features into directly-, indirectly-, and
un-controllable groups and perturbs only the controllable ones, while payload-padding attacks
[cite:prepadding2025] evade traffic classifiers by adding bytes that do not disturb transmission.

Our work is a direct response to the realism critique, and sharpens rather than contradicts it. We
do not claim a generic realizable evasion; we isolate one attack class — C2 beaconing — where the
detectable signal (timing/size regularity) is *exactly* a set of parameters the attacker already
controls for free, and show that there the evasion is not only practical but succeeds for 100% of
instances while a functional oracle confirms the beacon keeps working. The controllable-field idea
of [cite:controllability2026] appears here as the physical **timing-jitter** and
**size-jitter** axes; unlike feature-masking frameworks, our controllability is grounded in a
generator that must emit valid traffic. This also lets us measure quantities a feature-space attack
cannot define honestly: minimal *realizable* evading perturbation, queries-to-evasion against a
black-box verdict, and functional admissibility.

## 2.3 Detecting C2 beaconing

Flow-level detection of command-and-control beaconing keys on **timing regularity**: a beacon checks
in at a near-constant interval, and the standard statistical signal is a low **coefficient of
variation (CV)** of inter-arrival times, often combined with anomaly detectors
[cite:beaconing-cv2026], with surveys of C2 technique and defence going back a decade
[cite:c2-survey2014]. This is the detection premise we adopt (our `cv_iat` feature is precisely this
CV; Section 3.3) — and precisely the premise we attack. The beaconing-detection literature treats
timing regularity as a reliable discriminator; we show that because regularity is an attacker-set
parameter with functional slack, a detector that relies on it is defeated by a cost-free increase in
jitter, and that the near-identical *rigid benign* traffic (NTP, health checks, keepalives) bounds
how aggressively any such detector can tighten without false alarms.

## 2.4 Quantum machine learning for intrusion detection

A growing body of work applies quantum and quantum-classical models — variational quantum classifiers
(VQCs), quantum SVMs, quantum CNNs — to intrusion detection, generally reporting accuracy
competitive with or exceeding classical baselines on benchmark datasets
[cite:qmlids2024; cite:vqc-nisq2026; cite:qvcnn2024], within a broader QML-for-security literature
[cite:qml-sec-taxonomy2025]. This literature is almost entirely about *clean* accuracy; it does not
evaluate robustness to realizable evasion, and it typically works from static feature datasets. We
add the missing axis: a VQC and a classical MLP trained on **matched inputs** (Section 3.4) and
compared not on clean accuracy alone but on problem-space robustness to the same realized attack
traffic.

## 2.5 Adversarial robustness of quantum classifiers

Separately from the IDS setting, a quantum-ML-theory literature asks whether quantum classifiers are
*inherently* more robust. Early results showed VQCs are, like classical networks, vulnerable to
adversarial examples [cite:lu2020; cite:liao2021], but subsequent work raised the possibility of a
**quantum robustness advantage** — most prominently West et al. [cite:west-nature2023], who report
that quantum models can learn features classical attacks do not disturb, with scaling studies
[cite:west-benchmark2023] and a recent scoping review [cite:qml-robust-review2026] mapping the area.
Crucially, this entire literature evaluates robustness in **feature space** (L_p-bounded
perturbations of the encoded input), where "robustness" need not correspond to anything an attacker
could realize.

Our contribution to this strand is, to our knowledge, the first **problem-space** robustness
comparison of a quantum vs. classical classifier on a security task: robustness measured as
resistance to realized, functionality-preserving attack *traffic* rather than to L_p feature
perturbations. We find **no quantum robustness advantage** — the VQC is, if anything, marginally
easier to evade on the realistic task, and an apparent edge we observed on a saturated
(100%-clean-accuracy) version of the task disappears once detection is made realistic (Section 4.3).
This tempers the feature-space "quantum may be more robust" narrative with a problem-space data
point, in the one setting — actual traffic, functional constraint — that an operational defender
cares about.

## 2.6 Adversarial training and its limits

Adversarial training (folding adversarial examples into the training set) is the standard defense
and is studied for NIDS [cite:nids-challenges2024; cite:nids-advtrain2026], but it is known to
overfit to the perturbation *type* used during training and to generalize poorly to unseen or
transfer attacks. We instantiate this limitation concretely as a two-axis **arms race**: hardening
against the timing-jitter attack drives the jitter evasion rate to zero (at a measurable false-
positive cost) but leaves the previously-unused payload-size axis open, and the hardened detector is
re-broken for 100% of instances by size padding alone (Section 4.4). Because realizable traffic
offers several such functionally-free axes, single-axis adversarial training *relocates* the
vulnerability rather than closing it.

---

## References (working list — verify before LaTeX)

- [pierazzi2020] ✓ F. Pierazzi, F. Pendlebury, J. Cortellazzi, L. Cavallaro. "Intriguing Properties
  of Adversarial ML Attacks in the Problem Space." *IEEE Symposium on Security and Privacy (S&P)*,
  2020. arXiv:1911.02142.
- [fence2022] ✓ A. Chernikova, A. Oprea. "FENCE: Feasible Evasion Attacks on Neural Networks in
  Constrained Environments." *ACM Transactions on Privacy and Security (TOPS)* 25(4), Art. 34, 2022.
  arXiv:1909.10480.
- [apruzzese2022] ✓ G. Apruzzese, M. Andreolini, L. Ferretti, M. Marchetti, M. Colajanni. "Modeling
  Realistic Adversarial Attacks against Network Intrusion Detection Systems." *ACM Digital Threats:
  Research and Practice (DTRAP)* 3(3), 2022. arXiv:2106.09380.
- [elshehaby2023] ✓ M. elShehaby, A. Matrawy. "Evasion Adversarial Attacks Remain Impractical Against
  ML-based Network Intrusion Detection Systems, Especially Dynamic Ones." arXiv:2306.05494, 2023.
  [TODO: check for a final peer-reviewed venue.]
- [realistic-ps2024] ⚠ "Towards Realistic Problem-Space Adversarial Attacks against Machine Learning
  in Network Intrusion Detection." *ACM* (workshop), 2024. doi:10.1145/3664476.3669974. [TODO: authors.]
- [controllability2026] ⚠ "Controllability-Aware Adversarial Examples Against LLM-Based Network
  Traffic Classifiers." arXiv:2607.07739. [TODO: authors; confirm year/venue.]
- [prepadding2025] ⚠ "Adversarial Pre-Padding: Generating Evasive Network Traffic Against
  Transformer-Based Classifiers." arXiv:2510.25810. [TODO: authors.]
- [beaconing-cv2026] ⚠ "Beaconing detection via timing-based flow metadata analysis: implications for
  SCADA/ICS." *International Journal of Information Security (Springer)*, 2026.
  doi:10.1007/s10207-026-01335-w. [TODO: authors; confirm it uses CV of inter-arrival.]
- [c2-survey2014] ⚠ J. Gardiner, M. Cova, S. Nagaraja. "Command & Control: Understanding, Denying and
  Detecting." arXiv:1408.1136, 2014. [TODO: confirm author list.]
- [nids-survey2023] ⚠ "Adversarial Machine Learning for Network Intrusion Detection Systems: A
  Comprehensive Survey." *IEEE Communications Surveys & Tutorials*, 2023. [TODO: authors.]
- [nids-challenges2024] ⚠ "Adversarial Challenges in Network Intrusion Detection Systems: Research
  Insights and Future Prospects." arXiv:2409.18736, 2024. [TODO: authors.]
- [nids-understanding2026] ⚠ "Understanding the adversary: A survey of adversarial machine learning in
  network intrusion detection." *Computer Science Review / ScienceDirect*, 2026. [TODO: authors/venue.]
- [nids-advtrain2026] ⚠ "Enhancing Adversarial Robustness in Network Intrusion Detection: A
  Layer-wise Adaptive Regularization Approach." arXiv:2605.08910. [TODO: authors — or drop in favor
  of a stronger adversarial-training-limits cite.]
- [qmlids2024] ⚠ "QML-IDS: Quantum Machine Learning Intrusion Detection System." 2024. [TODO: authors/venue.]
- [vqc-nisq2026] ⚠ "Evaluation of Variational Quantum Classifiers (VQC) for Cyberattack Detection in
  the NISQ Era." *Journal of Sensor and Actuator Networks (JSAN)* 15(4):67, 2026. [TODO: authors.]
- [qvcnn2024] ⚠ "Network intrusion detection based on variational quantum convolution neural
  network." *The Journal of Supercomputing*, 2024. doi:10.1007/s11227-024-05919-y. [TODO: authors.]
- [qml-sec-taxonomy2025] ⚠ "Quantum Machine Learning for Cybersecurity: A Taxonomy and Future
  Directions." arXiv:2512.15286, 2025. [TODO: authors.]
- [lu2020] ✓ S. Lu, L.-M. Duan, D.-L. Deng. "Quantum Adversarial Machine Learning." *Physical Review
  Research* 2, 033212, 2020.
- [liao2021] ✓ H. Liao, I. Convy, W. J. Huggins, K. B. Whaley. "Robust in practice: Adversarial
  attacks on quantum machine learning." *Physical Review A* 103, 042427, 2021.
- [west-nature2023] ✓ M. T. West et al. "Towards quantum enhanced adversarial robustness in machine
  learning." *Nature Machine Intelligence*, 2023. [TODO: full author list.]
- [west-benchmark2023] ✓ M. T. West et al. "Benchmarking Adversarially Robust Quantum Machine
  Learning at Scale." *Physical Review Research* 5, 023186, 2023. arXiv:2211.12681.
- [qml-robust-review2026] ⚠ "Adversarial Robustness in Quantum Machine Learning: A Scoping Review."
  *Computers* 15(4):233, 2026. doi:10.3390/computers15040233. [TODO: authors.]
