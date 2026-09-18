"""Scenario runner / batch collector: builds the Mininet topology, drives benign + attack
traffic (optionally from several hosts per role, with randomized parameters), captures it, and
writes a labelled pcap + sidecar JSON per run for features.extract to consume. Repeat runs with
--runs to build a dataset. Linux + Mininet + root only — see README > Prerequisites.

Scenario schema (configs/scenarios/*.yaml):
  hosts: <N>                    total hosts (h1..hN); h1 is the victim/capture host by convention
  duration: <seconds>
  capture: {host: h1, iface: h1-eth0}
  servers: [{host, module, args}]        long-running (e.g. iperf server), started first
  attack:  [{hosts|host, class, module, args}]   each host runs `module`; flows labelled `class`
  benign:  [{hosts|host, module, args}]          each host runs `module`; flows labelled benign
An arg value may be a literal, "$hN" (resolved to that host's IP), {range: [lo, hi]} (sampled
per host per run) or {choice: [...]} (picked per host per run).
"""
import argparse
import json
import random
import time
from pathlib import Path

import yaml
from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink

from capture.capture import Capture
from topology.topo import IDSTopo


def _resolve_value(v, net, rng, self_host=None):
    if isinstance(v, str):
        if v == "$self_iface" and self_host is not None:
            return self_host.defaultIntf().name           # the running host's own interface
        if v.startswith("$"):
            return net.get(v[1:]).IP()
    if isinstance(v, dict):
        if "range" in v:
            lo, hi = v["range"]
            return rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) else round(rng.uniform(lo, hi), 3)
        if "choice" in v:
            return rng.choice(v["choice"])
    return v


def _resolve_args(args, net, rng, self_host=None):
    return {k: _resolve_value(v, net, rng, self_host) for k, v in args.items()}


def _cmd_for(module, args):
    cmd = ["python3", "-m", module]
    for k, v in args.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return cmd


def _hosts_of(flow):
    return flow["hosts"] if "hosts" in flow else [flow["host"]]


def _run_once(scenario, out_dir, run_id, rng):
    net = Mininet(topo=IDSTopo(n_hosts=scenario["hosts"]), controller=OVSController, link=TCLink)
    net.start()

    # Disable NIC offloads so captures record real MTU-sized packets, not GSO/TSO super-segments
    # (which would distort per-packet features). Needs ethtool.
    for h in net.hosts:
        for intf in h.intfList():
            if intf.name != "lo":
                h.cmd(f"ethtool -K {intf.name} gso off tso off gro off lro off")

    # Pre-populate ARP tables so hosts don't emit "who-has" resolution requests before their real
    # traffic — otherwise every attacker/benign host produces a stray 1-packet ARP flow that gets
    # labelled with its class, polluting the dataset (half the dos flows would be ARP noise). The
    # spoof attack is unaffected: arp_spoof emits its forged replies regardless of cache state.
    net.staticArp()

    try:
        cap_cfg = scenario["capture"]
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{scenario['name']}_{run_id:03d}"
        pcap_path = out / f"{stem}.pcap"
        cap_host = net.get(cap_cfg["host"])

        # Label every generator host by its role (see features/extract.py): attack hosts get the
        # entry's class, benign hosts get benign. Labelling is by MAC, robust to IP spoofing.
        mac_labels = {}
        for flow in scenario.get("attack", []):
            for h in _hosts_of(flow):
                mac_labels[net.get(h).MAC()] = flow["class"]
        for flow in scenario.get("benign", []):
            for h in _hosts_of(flow):
                mac_labels[net.get(h).MAC()] = "benign"

        servers = [net.get(s["host"]).popen(_cmd_for(s["module"], _resolve_args(s.get("args", {}), net, rng)))
                   for s in scenario.get("servers", [])]
        if servers:
            time.sleep(1.0)  # let servers bind before traffic starts

        with Capture(iface=cap_cfg["iface"], out_path=str(pcap_path), node=cap_host):
            procs = []
            for role in ("benign", "attack"):
                for flow in scenario.get(role, []):
                    for h in _hosts_of(flow):
                        host = net.get(h)
                        procs.append(host.popen(_cmd_for(flow["module"], _resolve_args(flow.get("args", {}), net, rng, self_host=host))))
            time.sleep(scenario["duration"])
            for p in procs:
                if p.poll() is None:
                    p.terminate()

        for s in servers:
            if s.poll() is None:
                s.terminate()

        (out / f"{stem}.json").write_text(json.dumps({
            "scenario": scenario["name"],
            "run_id": run_id,
            "mac_labels": mac_labels,
            "exclude_macs": [cap_host.MAC()],
        }, indent=2))
        print(f"wrote {pcap_path}  ({len(mac_labels)} labelled sources)")
    finally:
        net.stop()


def run_scenario(scenario_path, out_dir, runs=1, seed=0):
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)
    for i in range(runs):
        rng = random.Random(f"{scenario['name']}-{seed}-{i}")
        _run_once(scenario, out_dir, run_id=i, rng=rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default="capture/")
    parser.add_argument("--runs", type=int, default=1, help="repeat the scenario N times")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_scenario(args.scenario, args.out, runs=args.runs, seed=args.seed)


if __name__ == "__main__":
    main()
