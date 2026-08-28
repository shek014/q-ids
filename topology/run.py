"""Scenario runner: builds the Mininet topology, captures traffic on one host's
interface, launches the scenario's benign/attack generators on their hosts, and writes
a labelled pcap + sidecar JSON for features.extract to consume. Linux + Mininet + root
only — see README > Prerequisites. Not runnable on Windows.
"""
import argparse
import json
import time
from pathlib import Path

import yaml
from mininet.net import Mininet
from mininet.node import OVSController

from capture.capture import Capture
from topology.topo import IDSTopo


def _resolve_args(args, net):
    """Scenario args can reference another host's IP with "$hostname"."""
    resolved = {}
    for k, v in args.items():
        resolved[k] = net.get(v[1:]).IP() if isinstance(v, str) and v.startswith("$") else v
    return resolved


def _cmd_for(module, args):
    cmd = ["python3", "-m", module]
    for k, v in args.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return cmd


def run_scenario(scenario_path, out_dir):
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)

    net = Mininet(topo=IDSTopo(n_hosts=scenario["hosts"]), controller=OVSController)
    net.start()

    try:
        cap_cfg = scenario["capture"]
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pcap_path = out / f"{scenario['name']}.pcap"

        # Map each labelled host to its IP so extract.py can label flows by source (see
        # features/extract.py). Sources not listed default to benign at extraction time.
        ip_labels = {net.get(h).IP(): label for h, label in scenario.get("host_labels", {}).items()}

        # Long-running servers (e.g. iperf) must be up before clients connect.
        servers = [net.get(s["host"]).popen(_cmd_for(s["module"], _resolve_args(s.get("args", {}), net)))
                   for s in scenario.get("servers", [])]
        if servers:
            time.sleep(1.0)  # let servers bind before traffic starts

        with Capture(iface=cap_cfg["iface"], out_path=str(pcap_path)):
            procs = []
            for role in ("benign", "attack"):
                for flow in scenario.get(role, []):
                    host = net.get(flow["host"])
                    args = _resolve_args(flow.get("args", {}), net)
                    procs.append(host.popen(_cmd_for(flow["module"], args)))
            time.sleep(scenario["duration"])
            for p in procs:
                if p.poll() is None:
                    p.terminate()

        for s in servers:
            if s.poll() is None:
                s.terminate()

        (out / f"{scenario['name']}.json").write_text(json.dumps({
            "scenario": scenario["name"],
            "ip_labels": ip_labels,
        }, indent=2))
        print(f"wrote {pcap_path}  ({len(ip_labels)} labelled sources)")
    finally:
        net.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", default="capture/")
    args = parser.parse_args()
    run_scenario(args.scenario, args.out)


if __name__ == "__main__":
    main()
