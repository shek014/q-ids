"""pcap -> flow-level features, matching the schema in features/dataset.py. IPv4 only.
Expects each *.pcap in --pcap-dir to have a sidecar *.json with an "ip_labels" map
{<source ip>: <class name>}, as written by topology/run.py. Each flow is labelled by
its source: an attacker's IP yields attack flows, a benign host's IP yields benign flows,
so a single mixed capture produces correctly-labelled traffic of multiple classes. A source
not present in the map (e.g. the victim's own replies) defaults to benign.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scapy.all import ARP, ICMP, IP, TCP, UDP, PcapReader

from features.dataset import CLASS_NAMES, FEATURE_NAMES, save_dataset

PROTO_CODE = {"tcp": 0, "udp": 1, "icmp": 2, "arp": 3}


def _flow_key(pkt):
    if pkt.haslayer(ARP):
        return ("arp", pkt[ARP].psrc, None, pkt[ARP].pdst, None)
    if not pkt.haslayer(IP):
        return None
    if pkt.haslayer(TCP):
        return ("tcp", pkt[IP].src, pkt[TCP].sport, pkt[IP].dst, pkt[TCP].dport)
    if pkt.haslayer(UDP):
        return ("udp", pkt[IP].src, pkt[UDP].sport, pkt[IP].dst, pkt[UDP].dport)
    if pkt.haslayer(ICMP):
        return ("icmp", pkt[IP].src, None, pkt[IP].dst, None)
    return None


def parse_pcap(path):
    flows = defaultdict(list)
    for pkt in PcapReader(str(path)):
        key = _flow_key(pkt)
        if key is None:
            continue
        flags = str(pkt[TCP].flags) if pkt.haslayer(TCP) else ""
        flows[key].append({"ts": float(pkt.time), "length": len(pkt), "flags": flags})
    return flows


def _flow_features(key, packets):
    proto, src, sport, dst, dport = key
    ts = np.array(sorted(p["ts"] for p in packets))
    lengths = np.array([p["length"] for p in packets], dtype=float)
    duration = max(ts[-1] - ts[0], 1e-3)
    packet_count = len(packets)
    byte_count = lengths.sum()
    iat = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    syn = sum(1 for p in packets if "S" in p["flags"])
    ack = sum(1 for p in packets if "A" in p["flags"])
    fin = sum(1 for p in packets if "F" in p["flags"])
    rst = sum(1 for p in packets if "R" in p["flags"])

    return {
        "duration": duration, "packet_count": packet_count, "byte_count": byte_count,
        "mean_packet_size": lengths.mean(), "std_packet_size": lengths.std(),
        "packets_per_second": packet_count / duration, "bytes_per_second": byte_count / duration,
        "syn_count": syn, "ack_count": ack, "fin_count": fin, "rst_count": rst,
        "mean_iat": iat.mean(), "std_iat": iat.std(),
        "protocol": PROTO_CODE[proto],
        "_src": src, "_dst": dst, "_dport": dport, "_sport": sport,
    }


def extract_pcap(path, ip_labels, default_label="benign"):
    """Returns (X, y): feature rows and their per-flow labels. Each flow is labelled by its
    source IP via ip_labels; a source absent from the map falls back to default_label."""
    flows = parse_pcap(path)
    rows = [_flow_features(key, pkts) for key, pkts in flows.items()]

    dst_ports_by_src = defaultdict(set)
    src_ports_by_dst = defaultdict(set)
    for row in rows:
        if row["_dport"] is not None:
            dst_ports_by_src[row["_src"]].add(row["_dport"])
        if row["_sport"] is not None:
            src_ports_by_dst[row["_dst"]].add(row["_sport"])

    X, y = [], []
    for row in rows:
        row["unique_dst_ports"] = len(dst_ports_by_src[row["_src"]])
        row["unique_src_ports"] = len(src_ports_by_dst[row["_dst"]])
        label = ip_labels.get(row["_src"], default_label)
        if label not in CLASS_NAMES:
            continue  # unknown class name in the map — skip rather than mislabel
        X.append([row[name] for name in FEATURE_NAMES])
        y.append(CLASS_NAMES.index(label))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap-dir", default="capture")
    parser.add_argument("--out", default="data/dataset.npz")
    args = parser.parse_args()

    pcap_dir = Path(args.pcap_dir)
    X_parts, y_parts = [], []
    for pcap_path in sorted(pcap_dir.glob("*.pcap")):
        sidecar = pcap_path.with_suffix(".json")
        if not sidecar.exists():
            print(f"skipping {pcap_path.name}: no sidecar label file")
            continue
        ip_labels = json.loads(sidecar.read_text()).get("ip_labels", {})

        X, y = extract_pcap(pcap_path, ip_labels)
        if len(X) == 0:
            print(f"skipping {pcap_path.name}: no flows extracted")
            continue
        X_parts.append(X)
        y_parts.append(y)
        counts = {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES) if (y == i).any()}
        print(f"{pcap_path.name}: {len(X)} flows, {counts}")

    if not X_parts:
        raise SystemExit(f"no labelled pcaps found in {pcap_dir}")

    X, y = np.vstack(X_parts), np.concatenate(y_parts)
    save_dataset(args.out, X, y)
    print(f"wrote {len(X)} flows to {args.out}")


if __name__ == "__main__":
    main()
