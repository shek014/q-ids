"""Fabricates a flow-level dataset with the same schema features.extract.py produces from
real pcaps (see FEATURE_NAMES in features/dataset.py). Stand-in for real Mininet/Scapy/Nmap
captures so the rest of the pipeline (split, scale, train, evaluate) can be built and tested
before real traffic data is available. Swap for a real dataset by pointing --data at the
output of features.extract instead of this module's output.
"""
import argparse

import numpy as np

from features.dataset import CLASS_NAMES, save_dataset


def _clip_positive(x):
    return np.clip(x, 1e-3, None)


def _protocol(rng, n, weights):
    return rng.choice(4, size=n, p=weights)  # 0=tcp 1=udp 2=icmp 3=arp


def _pack(duration, packet_count, byte_count, syn, ack, fin, rst,
          unique_dst_ports, unique_src_ports, protocol, rng):
    duration = _clip_positive(duration)
    packet_count = np.clip(packet_count, 1, None)
    byte_count = np.clip(byte_count, packet_count * 40, None)  # min packet size ~40 bytes
    mean_pkt = byte_count / packet_count
    std_pkt = np.abs(rng.normal(mean_pkt * 0.15, mean_pkt * 0.05 + 1))
    pps = packet_count / duration
    bps = byte_count / duration
    mean_iat = duration / packet_count
    std_iat = np.abs(rng.normal(mean_iat * 0.4, mean_iat * 0.1 + 1e-4))

    return np.column_stack([
        duration, packet_count, byte_count, mean_pkt, std_pkt,
        pps, bps, syn, ack, fin, rst,
        mean_iat, std_iat, unique_dst_ports, unique_src_ports, protocol,
    ])


def generate_benign(n, rng):
    duration = rng.gamma(4.0, 1.2, size=n)
    packet_count = rng.poisson(40, size=n) + 5
    byte_count = packet_count * np.clip(rng.normal(450, 120, size=n), 60, 1500)
    syn = np.ones(n)
    ack = packet_count * 0.45 + rng.normal(0, 2, size=n)
    fin = np.ones(n)
    rst = rng.binomial(1, 0.05, size=n).astype(float)
    unique_dst_ports = rng.poisson(1.2, size=n) + 1
    unique_src_ports = rng.poisson(1.0, size=n) + 1
    protocol = _protocol(rng, n, [0.75, 0.20, 0.05, 0.0])
    return _pack(duration, packet_count, byte_count, syn, ack, fin, rst,
                 unique_dst_ports, unique_src_ports, protocol, rng)


def generate_dos(n, rng):
    duration = rng.gamma(2.0, 0.5, size=n)
    packet_count = rng.poisson(2000, size=n) + 200
    byte_count = packet_count * np.clip(rng.normal(70, 15, size=n), 40, 200)  # small flood packets
    syn = packet_count * np.clip(rng.normal(0.9, 0.05, size=n), 0.6, 1.0)
    ack = packet_count * np.clip(rng.normal(0.02, 0.02, size=n), 0, 0.2)
    fin = np.zeros(n)
    rst = packet_count * np.clip(rng.normal(0.05, 0.03, size=n), 0, 0.3)
    unique_dst_ports = rng.poisson(1.0, size=n) + 1  # single target port
    unique_src_ports = packet_count * np.clip(rng.normal(0.7, 0.1, size=n), 0.3, 1.0)  # spoofed sources
    protocol = _protocol(rng, n, [0.85, 0.15, 0.0, 0.0])
    return _pack(duration, packet_count, byte_count, syn, ack, fin, rst,
                 unique_dst_ports, unique_src_ports, protocol, rng)


def generate_recon(n, rng):
    duration = rng.gamma(2.0, 1.0, size=n)
    packet_count = rng.poisson(3, size=n) + 1
    byte_count = packet_count * np.clip(rng.normal(60, 10, size=n), 40, 200)
    syn = np.ones(n)
    ack = np.zeros(n)  # half-open scan, no completed handshake
    fin = np.zeros(n)
    rst = rng.binomial(1, 0.6, size=n).astype(float)  # closed-port RST responses
    unique_dst_ports = rng.poisson(50, size=n) + 10  # sweeping many ports
    unique_src_ports = rng.poisson(1.0, size=n) + 1
    protocol = _protocol(rng, n, [0.7, 0.3, 0.0, 0.0])
    return _pack(duration, packet_count, byte_count, syn, ack, fin, rst,
                 unique_dst_ports, unique_src_ports, protocol, rng)


def generate_spoof(n, rng):
    duration = rng.gamma(3.0, 1.0, size=n)
    packet_count = rng.poisson(20, size=n) + 5  # repeated gratuitous ARP replies
    byte_count = packet_count * np.clip(rng.normal(60, 5, size=n), 42, 90)
    syn = np.zeros(n)
    ack = np.zeros(n)
    fin = np.zeros(n)
    rst = np.zeros(n)
    unique_dst_ports = np.zeros(n)  # ARP has no ports
    unique_src_ports = np.zeros(n)
    protocol = np.full(n, 3)  # arp
    return _pack(duration, packet_count, byte_count, syn, ack, fin, rst,
                 unique_dst_ports, unique_src_ports, protocol, rng)


GENERATORS = {
    "benign": generate_benign,
    "dos": generate_dos,
    "recon": generate_recon,
    "spoof": generate_spoof,
}


def generate(n_samples, class_weights=None, seed=None):
    rng = np.random.default_rng(seed)
    if class_weights is None:
        class_weights = {"benign": 0.55, "dos": 0.20, "recon": 0.15, "spoof": 0.10}

    X_parts, y_parts = [], []
    for label_idx, name in enumerate(CLASS_NAMES):
        n = int(round(n_samples * class_weights[name]))
        X_parts.append(GENERATORS[name](n, rng))
        y_parts.append(np.full(n, label_idx))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/dataset.npz")
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = generate(args.n_samples, seed=args.seed)
    save_dataset(args.out, X, y)
    print(f"wrote {len(X)} samples to {args.out}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y == i).sum()}")


if __name__ == "__main__":
    main()
