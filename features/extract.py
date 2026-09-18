"""pcap -> flow-level features, matching the schema in features/dataset.py. IPv4 only.
Expects each *.pcap in --pcap-dir to have a sidecar *.json with a "mac_labels" map
{<source MAC>: <class name>} and optional "exclude_macs", as written by topology/run.py.

Flow definition: flows are keyed by (protocol, source MAC, destination) — the sender's real
hardware address plus the target host. NOT the 5-tuple, because the attacks spoof the fields a
5-tuple keys on: a SYN flood randomizes the source IP and port on every packet, so under
5-tuple grouping one flood explodes into thousands of meaningless 1-packet flows; ARP poisoning
forges the ARP psrc. Keying by the sender's MAC (which the generators never spoof) collapses a
flood into ONE flow whose aggregate features actually describe a flood, and likewise a port scan
(many destination ports from one sender) into one flow. Port diversity then becomes a within-flow
signal: unique_src_ports is high for a spoofed flood, unique_dst_ports is high for a scan.

Each flow is labelled by its source MAC via mac_labels; a sender not in the map defaults to
benign, and a sender in exclude_macs (e.g. the capture/victim host) is dropped entirely.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scapy.all import ARP, ICMP, IP, TCP, UDP, Ether, PcapReader

from features.dataset import CLASS_NAMES, FEATURE_NAMES, save_dataset

PROTO_CODE = {"tcp": 0, "udp": 1, "icmp": 2, "arp": 3}


def _packet(pkt):
    """Returns (flow_key, packet_info) or None. flow_key = (protocol, source MAC, destination)."""
    src_mac = pkt[Ether].src.lower() if pkt.haslayer(Ether) else None
    length = pkt.wirelen if getattr(pkt, "wirelen", None) else len(pkt)
    info = {"ts": float(pkt.time), "length": length, "flags": "",
            "sport": None, "dport": None, "src_mac": src_mac}

    if pkt.haslayer(ARP):
        return ("arp", src_mac, pkt[ARP].pdst), info
    if not pkt.haslayer(IP):
        return None
    dst = pkt[IP].dst
    if pkt.haslayer(TCP):
        info["flags"] = str(pkt[TCP].flags)
        info["sport"], info["dport"] = int(pkt[TCP].sport), int(pkt[TCP].dport)
        return ("tcp", src_mac, dst), info
    if pkt.haslayer(UDP):
        info["sport"], info["dport"] = int(pkt[UDP].sport), int(pkt[UDP].dport)
        return ("udp", src_mac, dst), info
    if pkt.haslayer(ICMP):
        return ("icmp", src_mac, dst), info
    return None


def parse_pcap(path):
    flows = defaultdict(list)
    for pkt in PcapReader(str(path)):
        r = _packet(pkt)
        if r is None:
            continue
        key, info = r
        flows[key].append(info)
    return flows


def _flow_features(key, packets):
    proto = key[0]
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
    # port diversity within the flow: high unique_src_ports = spoofed flood, high unique_dst_ports = scan
    dports = {p["dport"] for p in packets if p["dport"] is not None}
    sports = {p["sport"] for p in packets if p["sport"] is not None}

    return {
        "duration": duration, "packet_count": packet_count, "byte_count": byte_count,
        "mean_packet_size": lengths.mean(), "std_packet_size": lengths.std(),
        "packets_per_second": packet_count / duration, "bytes_per_second": byte_count / duration,
        "syn_count": syn, "ack_count": ack, "fin_count": fin, "rst_count": rst,
        "mean_iat": iat.mean(), "std_iat": iat.std(),
        "unique_dst_ports": len(dports), "unique_src_ports": len(sports),
        "protocol": PROTO_CODE[proto],
        "_src_mac": packets[0]["src_mac"],
    }


def extract_pcap(path, mac_labels, exclude_macs=(), default_label="benign"):
    """Returns (X, y): one feature row per (protocol, source MAC, destination) flow, labelled by
    source MAC via mac_labels (keys lower-cased); senders absent from the map fall back to
    default_label, and flows whose source MAC is in exclude_macs are dropped entirely."""
    mac_labels = {k.lower(): v for k, v in mac_labels.items()}
    exclude = {m.lower() for m in exclude_macs}
    flows = parse_pcap(path)

    X, y = [], []
    for key, pkts in flows.items():
        row = _flow_features(key, pkts)
        if row["_src_mac"] in exclude:
            continue  # drop the capture/victim host's own traffic
        label = mac_labels.get(row["_src_mac"], default_label)
        if label not in CLASS_NAMES:
            continue  # unknown class name in the map — skip rather than mislabel
        X.append([row[name] for name in FEATURE_NAMES])
        y.append(CLASS_NAMES.index(label))
    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)
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
        sidecar_data = json.loads(sidecar.read_text())
        mac_labels = sidecar_data.get("mac_labels", {})
        exclude_macs = sidecar_data.get("exclude_macs", [])

        X, y = extract_pcap(pcap_path, mac_labels, exclude_macs=exclude_macs)
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
