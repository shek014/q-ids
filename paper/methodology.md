# 3. Methodology

> Draft — Methodology section. Written for the paper (reviewers / co-author). Numbers reflect the
> realistic-benign dataset (2026-10); `[cite]` marks a reference to add; `[TODO]` marks a decision
> to confirm with the co-author.

We evaluate the robustness of flow-level machine-learning intrusion detectors to *realizable*
(problem-space) adversarial evasion, and compare a classical neural network against a
quantum-classical classifier under identical conditions. This section describes the threat model
(3.1), the emulated testbed and traffic (3.2), the flow representation (3.3), the two detectors and
the matched-input protocol that makes their comparison fair (3.4), the problem-space evasion method
and its functional oracle (3.5), the paired transferability protocol (3.6), and the
adversarial-training defense and its escalation (3.7). All code and captured data are released for
reproducibility [cite repo/Zenodo DOI].

## 3.1 Threat model

We consider an attacker who has compromised an internal host and operates a command-and-control (C2)
beacon: the host periodically "checks in" to an attacker-controlled server to receive commands. A
network intrusion detection system (NIDS) observes traffic at the flow level and classifies each
flow as benign or C2.

The attacker's goal is to evade detection while **preserving functionality** — the beacon must keep
maintaining its control channel. The attacker:

- **can** manipulate the realizable parameters of their own traffic: the check-in interval, the
  timing jitter (randomness of the interval), the payload size, and the size jitter (randomness of
  the payload size). These are degrees of freedom a real C2 implementation controls and that do not,
  within bounds, impair the beacon;
- **cannot** alter other hosts' traffic, the detector's parameters, or the feature-extraction
  pipeline, and cannot submit feature vectors directly — every change must be realized as actual
  traffic on the wire;
- **can** query the detector in a black-box manner (observe benign/malicious verdicts), modelling an
  attacker who tests candidate configurations against a deployed or surrogate detector.

This is a **problem-space** threat model [cite Pierazzi et al.]: perturbations are constrained to
what is physically realizable and functionality-preserving, in contrast to **feature-space** attacks
that perturb the feature vector directly and may correspond to no realizable traffic. A perturbation
is *admissible* only if the resulting beacon remains functional (Section 3.5).

## 3.2 Testbed and traffic generation

Problem-space evasion requires *controllable* traffic generation — static benchmark datasets cannot
be manipulated in the loop — so we emulate a network in **Mininet** [cite]. Hosts attach to a single
Open vSwitch switch via `TCLink` links capped at 10 Mbit/s, which bounds capture volume to a
realistic segment rate. On every host we disable NIC offloads (`gso`, `tso`, `gro`, `lro`) so the
capture records real MTU-sized packets rather than offload super-segments, and we pre-populate ARP
tables (`staticArp`) so address-resolution traffic does not appear as stray flows. Traffic is
captured with `tcpdump` inside the relevant host's network namespace at a 128-byte snap length
(headers only); per-packet size features use the recorded wire length rather than the truncated
capture length.

Each scenario runs for 60 s and is repeated with per-host, per-run randomized parameters to produce
a dataset. One host (h1) acts as the victim/collector and runs the services; the remaining hosts
generate traffic in one of the roles below.

**C2 beacons (malicious).** A beacon sends a single UDP datagram check-in to a C2 endpoint every
`interval` seconds (± `jitter`), with payload size `payload_size` (± `size_jitter`), and reads back
a reply. Modelling check-ins as single datagrams (DNS/heartbeat-style C2) means a flow's packet
inter-arrival times *are* its check-in intervals, so timing regularity is expressed directly in the
flow features rather than being obscured by connection-level mechanics. A naive beacon is rigid: low
`jitter` and low `size_jitter`.

**Benign traffic.** Crucially for a realistic detection task, benign traffic spans the full range of
*regularity*. Real networks contain highly regular benign traffic — NTP, health checks, keepalives,
telemetry — that is nearly indistinguishable from a beacon at the flow level, which is precisely why
flow-based C2 detection is hard. We generate: (a) **rigid** benign periodic traffic in the *same*
timing/size band as the beacons (NTP/strict-health-check analogues); (b) moderately regular
keepalives; (c) loose, irregular polling; (d) bursty short TCP application sessions across several
ports, some failing; and (e) bulk transfer. The rigid benign traffic (a) is a deliberate confuser:
the detector cannot perfectly separate it from a beacon, which keeps clean accuracy realistic rather
than saturated (Section 4).

The beacons and the periodic benign traffic (a–c) are produced by the *same* generator differing
only in the jitter/size parameter regime; the role (and hence label) is assigned per host. The
contested boundary is therefore near-boundary by construction: the only systematic difference
between a beacon and a rigid benign service is the statistics of its timing and size — exactly the
axes the evasion will manipulate.

## 3.3 Flow features and labeling

Packets are aggregated into flows keyed by the tuple **(protocol, source MAC, destination)** rather
than the 5-tuple. The 5-tuple is unsuitable here because an attacker can vary source ports and
spoof addresses, which would fragment one logical activity into many single-packet "flows"; keying
on the sender's hardware address — which the generators never spoof — collapses each host's activity
into one flow whose aggregate statistics are meaningful, and makes port diversity a *within-flow*
signal. Each flow is labeled by the role of its source MAC; the collector/victim host's own traffic
is excluded.

From each flow we extract 17 flow-level features: `duration`, `packet_count`, `byte_count`,
`mean_packet_size`, `std_packet_size`, `packets_per_second`, `bytes_per_second`, the TCP flag counts
`syn/ack/fin/rst_count`, `mean_iat`, `std_iat`, `cv_iat`, `unique_dst_ports`, `unique_src_ports`, and
a categorical `protocol`. Here `iat` denotes inter-arrival time and
**`cv_iat = std_iat / mean_iat`** is the coefficient of variation of the inter-arrival times — a
scale-invariant measure of timing regularity. `cv_iat` is central to this study: a rigid beacon has
low `cv_iat` at any interval, whereas naturally-jittered benign traffic has high `cv_iat`, and
(unlike raw `std_iat`) it does not conflate regularity with interval length.

## 3.4 Detectors and the matched-input protocol

We compare two detectors on an identical binary task (benign vs. C2).

**Classical MLP.** A multilayer perceptron implemented from scratch in NumPy (no ML frameworks):
two hidden layers of 32 and 16 units with ReLU activations and a 2-way softmax output, trained by
mini-batch gradient descent with the Adam optimizer (learning rate 0.01, batch size 64, up to 150
epochs with early stopping, patience 20) under class-balanced loss weighting. The softmax +
cross-entropy gradient was verified against finite differences.

**Quantum-classical VQC.** A variational quantum classifier implemented in PennyLane [cite]: an
`AngleEmbedding` of the (PCA-reduced) features onto 8 qubits, followed by 4 `StronglyEntanglingLayers`,
reading out one Pauli-Z expectation per class from the first two wires, passed through a softmax and
trained with cross-entropy (learning rate 0.05, batch size 16, up to 30 epochs, patience 8). Training
uses exact gradients via the `default.qubit` state-vector simulator.

**Matched inputs.** To make the comparison attributable to the model rather than the information it
receives, both detectors are trained on *identical* inputs for the comparison: features are
standardized (z-score, fit on the training split only) and reduced by PCA to the 8-component qubit
budget; for the VQC the components are additionally rescaled into the angle range required by
`AngleEmbedding`, a monotone per-feature transform that carries no additional information. We also
train the MLP on the full 17-dimensional ("native") feature set to characterize the detector a
defender would actually deploy. All preprocessing transforms are fit on the training split only.

The dataset comprises 420 flows (300 benign, 120 C2) split 70/15/15 into train/validation/test,
stratified by class.

## 3.5 Problem-space evasion

The evasion is a model-in-the-loop search for the smallest realizable, functionality-preserving
perturbation that flips the detector's verdict to benign.

**Query primitive.** A single query takes a parameter vector θ = (interval, jitter, payload_size,
size_jitter), generates a *real* beacon with those parameters in a short Mininet capture (a minimal
two-host topology; beacon flow features are per-source-MAC and therefore independent of background
traffic), extracts the beacon's flow, and returns its 17-feature vector. The capture window matches
the 60 s training window so query flows are in-distribution with the trained detector.

**Functional oracle.** During each query the beacon logs the timestamp of every *successful*
(round-trip-confirmed) check-in. The beacon is deemed **functional** iff it completed at least two
successful check-ins and no gap between consecutive successful check-ins exceeded a maximum tolerable
control-latency `MAX_INTERVAL` (30 s). This operationalizes "the attack still works": a perturbation
that silences the beacon or stretches its gaps past the operator's tolerance is inadmissible.

**Line search.** For a given detector and a given realizable axis, we fix the other parameters and
sweep the axis over an ascending grid. For each beacon instance (seed) we find the **minimal value
at which the detector assigns P(benign) ≥ 0.5 while the beacon remains functional** — the minimal
evading perturbation — and record the number of queries spent reaching it (queries-to-evasion). We
report evasion success rate, the distribution of minimal evading perturbations, and
queries-to-evasion across instances. We search the **timing-jitter** axis first (Section 4), and the
**payload-size-jitter** axis for the defense escalation (Section 3.7); both are functionally free in
the sense that neither prevents a check-in.

## 3.6 Paired transferability

To compare the two detectors on *identical* traffic and to measure cross-model transfer, we decouple
capture from scoring. A **capture-once sweep** runs the full jitter grid for every beacon instance a
single time, saving each flow's features and functional verdict with no detector in the loop. These
flows are then scored **offline against both detectors**, yielding (a) a *paired* comparison of
minimal evading jitter on the same flows, and (b) transferability: whether a beacon tuned to just
evade one detector also evades the other. Because the captured flows are a property of the attack and
not of any detector, the same sweep is reused to evaluate every (including adversarially-trained)
model, which also removes capture variance from the comparison.

## 3.7 Adversarial-training defense and axis escalation

Finally we test whether adversarial training restores robustness. We fold evaded beacons from the
sweep into the training set labeled as C2 and retrain each detector; to avoid leakage, only a subset
of beacon instances (seeds) is used for training, with the remainder held out for evaluation. We
report the hardened detectors' clean accuracy and, in particular, their **benign false-positive
rate**, since teaching the detector that jittered beacons are malicious risks flagging the
near-identical rigid benign traffic.

We then evaluate the hardened detectors in two steps. First, we re-score the held-out jitter
evasions to test whether the original attack still succeeds. Second — because the jitter defense can
only exploit the *timing* axis — we mount an **axis escalation**: holding timing jitter at an
already-evading value, we run the line search on the previously-unused **payload-size-jitter** axis
against the hardened detector. This tests whether a defender who hardens against one realizable axis
has closed the vulnerability or merely displaced it to another equally-free axis.
