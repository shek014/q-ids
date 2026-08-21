# q-ids

Comparative analysis of classical and quantum-classical ML for network intrusion detection under realistic adversarial attacks.

> **Status: early scaffold.** This README describes the intended design. The pipeline is not implemented yet — see [Roadmap](#roadmap) for what exists and what doesn't.

## Overview

`q-ids` builds two intrusion detection classifiers over the *same* network traffic dataset and compares them head to head:

1. **Classical** — a feed-forward neural network implemented directly in NumPy (forward pass, backprop, and optimizer written out, no framework).
2. **Quantum-classical** — a variational quantum classifier (VQC) built with PennyLane, with classical pre-processing feeding a parameterized quantum circuit.

The dataset is not downloaded from a public corpus. It is *generated* in a controlled problem-space environment: a virtual network is emulated with Mininet, benign and attack traffic are driven through it with Scapy and Nmap, packets are captured, and flow-level features are extracted from the captures. This keeps the ground-truth labels exact and makes the attack mix reproducible.

The question the project is built to answer: **on identical, realistically generated traffic, does a quantum model buy anything over a classical model of comparable capacity — in accuracy, in sample efficiency, or in robustness to evasive attack traffic?**

## Pipeline

```
┌─────────────────┐
│  Mininet topo   │  emulated hosts, switches, links
└────────┬────────┘
         │
    ┌────┴─────┐
    │  Traffic │  benign: iperf, curl, ping, synthetic app flows
    │   drive  │  attack: Nmap scans, Scapy-crafted floods/spoofing
    └────┬─────┘
         │  tcpdump on the switch / host interfaces
         ▼
   ┌───────────┐
   │  .pcap    │  raw capture, per-scenario
   └─────┬─────┘
         │  flow reassembly + feature extraction
         ▼
   ┌───────────┐
   │  dataset  │  labelled feature vectors (CSV / .npz)
   └─────┬─────┘
         │  split, scale, reduce dimensionality
         ├──────────────────────┬──────────────────────┐
         ▼                      ▼                      │
  ┌─────────────┐        ┌─────────────┐               │
  │  NumPy MLP  │        │ PennyLane   │               │
  │ (classical) │        │   VQC       │               │
  └──────┬──────┘        └──────┬──────┘               │
         └───────────┬──────────┘                      │
                     ▼                                 │
              ┌─────────────┐                          │
              │ evaluation  │ ◄────────────────────────┘
              └─────────────┘  metrics, curves, cost accounting
```

## Tech stack

| Layer | Tool | Role |
|---|---|---|
| Network emulation | **Mininet** | Virtual topology of hosts/switches; the problem-space environment |
| Attack traffic | **Scapy** | Packet crafting — SYN floods, spoofed sources, malformed headers, evasive timing |
| Reconnaissance | **Nmap** | Port scans (SYN/FIN/XMAS/UDP), service and version detection |
| Capture | **tcpdump / libpcap** | Raw packet capture per scenario |
| Feature extraction | **Scapy / Python** | Flow reassembly and flow-level statistics |
| Classical model | **Python + NumPy** | MLP written from scratch — layers, backprop, SGD/Adam |
| Quantum model | **PennyLane** | Variational circuits, angle embedding, parameter-shift gradients, `default.qubit` simulator |
| Evaluation | **NumPy + Matplotlib** | Metrics, confusion matrices, ROC/PR curves, training-cost comparison |

Deliberately **not** used for the models themselves: scikit-learn, PyTorch, TensorFlow. The classical baseline is hand-written so that the classical and quantum sides are compared on equal, fully-visible footing rather than through differently-optimized library internals. Utility libraries (pandas for wrangling, matplotlib for plots) are fine.

## Threat model / attack classes

The generated dataset is intended to cover, at minimum:

- **Reconnaissance** — TCP SYN scan, FIN/XMAS/NULL scans, UDP scan, service enumeration (Nmap)
- **Denial of service** — SYN flood, UDP flood, ICMP flood, slow-rate exhaustion (Scapy)
- **Spoofing** — source-address spoofing, ARP poisoning within the emulated LAN
- **Evasive variants** — the "realistic adversarial" part: fragmented payloads, decoy addresses, timing-templated slow scans, and packet-rate shaping that puts attack flows inside the benign statistical envelope

The evasive variants matter most. A model that separates a full-rate SYN flood from a `ping` is not saying much; the comparison is only interesting near the decision boundary.

## The dimensionality constraint

This is the central design tension, and it shapes the whole experiment.

Simulated quantum circuits scale exponentially in qubit count, so a VQC realistically handles a small number of features — on the order of 4 to 12 with angle embedding, one feature per qubit. Flow-level IDS feature sets are typically much wider than that.

So the two models are compared under two regimes:

- **Matched-input** — both models see the same reduced feature set (PCA or ranked feature selection down to *n* = qubit count). This isolates the model, and is the primary comparison.
- **Native-input** — the classical model additionally gets the full feature set, as an upper reference point for what is being given up.

Any claim about quantum advantage that only holds in the matched-input regime should be reported as exactly that, and the parameter counts and wall-clock training cost of both models are reported alongside accuracy. A VQC that ties a classical MLP while taking orders of magnitude longer to train on a simulator has not won anything, and the results tables are structured to make that visible rather than to hide it.

## Planned repository layout

```
q-ids/
├── topology/          Mininet topology definitions and scenario runners
├── traffic/
│   ├── benign/        benign traffic generators
│   └── attack/        Scapy attack scripts, Nmap scan drivers
├── capture/           tcpdump orchestration, raw pcap output (gitignored)
├── features/          flow reassembly, feature extraction, dataset assembly
├── data/              generated datasets (gitignored; regenerate from scripts)
├── models/
│   ├── classical/     NumPy MLP — layers, losses, optimizers
│   └── quantum/       PennyLane VQC — embeddings, ansätze, training loop
├── evaluation/        metrics, plots, comparison tables
├── configs/           scenario and experiment configuration
└── results/           committed figures and metric dumps
```

Datasets and pcaps stay out of version control — the generation scripts plus a seeded config are the reproducible artifact, not the bytes.

## Prerequisites

**Mininet requires Linux.** It depends on network namespaces and Open vSwitch and does not run natively on Windows or macOS. Options for the data-generation half of the pipeline:

- a Linux VM (the Mininet project publishes a prebuilt image),
- WSL2 with a systemd-enabled distro and Open vSwitch installed, or
- a Linux host / container with `--privileged` and `NET_ADMIN`.

Model training and evaluation are pure Python and run anywhere. A practical split is: generate captures on Linux, commit or copy the extracted feature CSVs, and train wherever is convenient.

Traffic generation needs root — Scapy raw sockets, Nmap SYN scans, and tcpdump all require elevated privileges. **Only run the attack scripts against the emulated Mininet hosts.** They target RFC 1918 addresses inside the virtual topology; pointing them at anything else is out of scope for this project and likely illegal.

## Setup

```bash
git clone https://github.com/shek014/q-ids.git
cd q-ids

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Mininet and Nmap are system packages, installed outside the virtualenv:

```bash
sudo apt install mininet openvswitch-switch nmap tcpdump
```

## Usage

Intended entry points, once implemented:

```bash
# 1. bring up the topology and run a labelled traffic scenario
sudo python -m topology.run --scenario configs/scenarios/synflood.yaml --out capture/

# 2. extract flow features from the captures into a dataset
python -m features.extract --pcap-dir capture/ --out data/dataset.npz

# 3. train both models on the same split
python -m models.classical.train --data data/dataset.npz --config configs/mlp.yaml
python -m models.quantum.train   --data data/dataset.npz --config configs/vqc.yaml

# 4. compare
python -m evaluation.compare --runs results/ --out results/comparison/
```

## Evaluation

Reported for every model, per attack class and in aggregate:

- Accuracy, precision, recall, F1
- **False positive rate** — weighted heavily; an IDS that cries wolf is unusable regardless of its recall
- ROC-AUC and precision-recall AUC (PR-AUC is the more honest curve here, since attack traffic is the minority class)
- Confusion matrix across attack classes
- Learning curves against training-set size — the sample-efficiency claim often made for quantum models is tested directly here
- Parameter count, circuit depth / qubit count, and wall-clock training time

Splits are stratified and, where flows come from the same scenario run, grouped so that packets from a single flow cannot straddle the train/test boundary — otherwise the numbers are inflated by leakage.

## Non-goals

To keep the comparison honest, this project explicitly does **not** claim or attempt:

- **Quantum advantage.** Nothing here runs on real quantum hardware, and no asymptotic speedup is being demonstrated or implied. `default.qubit` is a classical simulator; the VQC is compared as an alternative model architecture, not as a faster computer.
- **A NISQ-hardware deployment story.** No noise models, no hardware-specific transpilation, no results from IBMQ/IonQ/etc. If that's added later it's a separate, explicitly labelled experiment, not folded into the main comparison.
- **State-of-the-art IDS accuracy.** The traffic is self-generated in an emulated topology, not a benchmark dataset (CIC-IDS2017, NSL-KDD, etc.), so results are not directly comparable to published leaderboard numbers. The point is a controlled, apples-to-apples comparison between two models on identical data, not chasing a top score.
- **General-purpose intrusion detection.** The attack set is scoped to what's listed under [Threat model](#threat-model--attack-classes); this is not claimed to generalize to attack classes outside it (e.g. application-layer exploits, malware C2 traffic).

## Roadmap

- [ ] Mininet topology definitions and scenario runner
- [ ] Benign traffic generators
- [ ] Scapy attack scripts (recon, DoS, spoofing)
- [ ] Nmap scan drivers
- [ ] Capture orchestration
- [ ] Flow reassembly and feature extraction
- [ ] Dataset assembly with stratified, leakage-safe splits
- [ ] NumPy MLP — layers, backprop, optimizers
- [ ] PennyLane VQC — embedding, ansatz, training loop
- [ ] Evaluation harness and comparison tables
- [ ] Evasive / adversarial attack variants
- [ ] Results write-up

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Abhishek Ravicharan.
