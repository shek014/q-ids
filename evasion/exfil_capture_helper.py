"""Capture-only half of the EXFIL evasion query (PROTOTYPE), run under SYSTEM python3 (which has
Mininet but no numpy). Runs ONE data-exfiltration transfer with the given parameters to an
exfil_sink, captures the traffic, and writes the pcap + a meta sidecar + the delivery-progress log.

Topology (--topology):
  full    (default) the training scenario's own network (configs/scenarios/exfil.yaml): same hosts,
          servers and background traffic as the dataset captures, so the exfil shares the 10 Mbit/s
          link exactly as it did in training and the query flow is in-distribution. Background
          parameters are sampled once per seed, so every throttle level of an instance sees the same
          background. The controlled exfil runs on h2 to its own collector port (EXFIL_QUERY_PORT,
          logged) so the delivery check only counts its bytes.
  minimal the original 2-host topology (no contention). Kept so older sweeps stay reproducible; its
          naive flows run faster and more regularly than any training exfil (out of distribution). It does NO feature extraction (that needs numpy/scapy and happens in the
.venv half, evasion/exfil_query.py), so this file imports nothing beyond Mininet + the pure-stdlib
capture wrapper.

Mirrors evasion/capture_helper.py (the C2 version). Invoked as a subprocess by evasion/exfil_query.py;
not meant to be run by hand.
"""
import argparse
import json
import time
from pathlib import Path

from mininet.net import Mininet
from mininet.node import OVSController
from mininet.link import TCLink

from capture.capture import Capture
from topology.topo import IDSTopo

EXFIL_PORT = 8443          # collector port (minimal topology; benign transfers in the full one)
EXFIL_QUERY_PORT = 8444    # the controlled exfil's own logged collector in the full topology
DEFAULT_SCENARIO = "configs/scenarios/exfil.yaml"


def _start_full(scenario_path, seed, recv_path):
    """Build the training scenario's network and start everything EXCEPT the controlled exfil (h2).
    Returns (net, h1, h2, procs_to_stop, background_macs)."""
    import random
    import yaml
    from topology.run import _resolve_args, _cmd_for, _hosts_of
    sc = yaml.safe_load(open(scenario_path))
    rng = random.Random(f"exfil-query-bg-{seed}")
    net = Mininet(topo=IDSTopo(n_hosts=sc["hosts"]), controller=OVSController, link=TCLink)
    net.start()
    for h in net.hosts:
        for intf in h.intfList():
            if intf.name != "lo":
                h.cmd(f"ethtool -K {intf.name} gso off tso off gro off lro off")
    net.staticArp()
    h1, h2 = net.get("h1"), net.get("h2")
    servers = [net.get(sv["host"]).popen(_cmd_for(sv["module"], _resolve_args(sv.get("args", {}), net, rng)))
               for sv in sc.get("servers", [])]
    servers.append(h1.popen(["python3", "-m", "traffic.exfil_sink",
                             "--ports", str(EXFIL_QUERY_PORT), "--log", str(recv_path)]))
    background, bg_macs = [], []
    for role in ("benign", "attack"):
        for flow in sc.get(role, []):
            for hn in _hosts_of(flow):
                if hn == "h2":
                    continue                      # h2 is the controlled exfil
                host = net.get(hn)
                background.append((host, _cmd_for(flow["module"],
                                                  _resolve_args(flow.get("args", {}), net, rng, self_host=host))))
                bg_macs.append(host.MAC())
    return net, h1, h2, servers, background, bg_macs


def capture(theta, window, seed, out_dir, topology="full", scenario=DEFAULT_SCENARIO):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcap_path = out / "query.pcap"
    log_path = out / "progress.log"
    recv_path = out / "received.log"     # collector-side received totals, for the delivery cross-check
    for p in (log_path, recv_path):
        if p.exists():
            p.unlink()

    if topology == "full":
        net, h1, h2, servers, background, bg_macs = _start_full(scenario, seed, recv_path)
        port = EXFIL_QUERY_PORT
    else:
        net = Mininet(topo=IDSTopo(n_hosts=2), controller=OVSController, link=TCLink)
        net.start()
        h1, h2 = net.get("h1"), net.get("h2")
        for h in (h1, h2):
            for intf in h.intfList():
                if intf.name != "lo":
                    h.cmd(f"ethtool -K {intf.name} gso off tso off gro off lro off")
        net.staticArp()
        servers = [h1.popen(["python3", "-m", "traffic.exfil_sink",
                             "--ports", str(EXFIL_PORT), "--log", str(recv_path)])]
        background, bg_macs, port = [], [], EXFIL_PORT
    bg_procs = []
    try:
        time.sleep(1.0)                                  # let servers bind

        exfil_cmd = [
            "python3", "-m", "traffic.exfil_client",
            "--target", h1.IP(), "--port", str(port),
            "--volume", str(int(theta["volume"])),
            "--rate", str(theta["rate"]),                        # already the EFFECTIVE (throttled) rate
            "--chunk-size", str(int(theta.get("chunk_size", 4096))),
            "--jitter", str(theta.get("jitter", 0.0)),
            "--deadline", str(theta.get("deadline", window)),
            "--seed", str(seed),
            "--log", str(log_path),
        ]
        with Capture(iface="h1-eth0", out_path=str(pcap_path), node=h1):
            # background starts with the exfil, as in a training run (all generators launch together)
            bg_procs = [host.popen(cmd) for host, cmd in background]
            t0 = time.time()
            client = h2.popen(exfil_cmd)
            # adaptive wait: once the client exits (payload delivered, or it hit its own deadline) the
            # exfil flow is over — no further packets come from it — so capturing the rest of the window
            # only adds idle time. Poll until it exits, capped at window+1 (the client self-stops at its
            # deadline, which is <= window), then drain briefly so the final packets land in the pcap.
            # Flow features are identical to sleeping the full window; this just skips the idle tail for
            # fast (low-throttle) transfers, which dominate the grid.
            deadline_cap = t0 + window + 1.0
            while client.poll() is None and time.time() < deadline_cap:
                time.sleep(0.25)
            if client.poll() is None:
                client.terminate()
            time.sleep(1.0)                              # drain trailing packets into the capture
            for bp in bg_procs:
                if bp.poll() is None:
                    bp.terminate()
        # give the collector a moment to flush the final per-connection total (it logs on connection
        # EOF) before tearing the sink down, so the receiver-side delivery cross-check sees it
        for _ in range(30):                              # up to ~3s
            if recv_path.exists() and recv_path.stat().st_size > 0:
                break
            time.sleep(0.1)
        for sv in servers:
            if sv.poll() is None:
                sv.terminate()

        (out / "meta.json").write_text(json.dumps({
            "exfil_mac": h2.MAC(), "capture_mac": h1.MAC(), "background_macs": bg_macs,
            "topology": topology,
            "window_start": t0, "window": window, "theta": theta, "seed": seed,
        }))
    finally:
        for bp in bg_procs:
            if bp.poll() is None:
                bp.terminate()
        net.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume", type=int, required=True)
    p.add_argument("--rate", type=float, required=True, help="effective (post-throttle) bytes/sec")
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--window", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--topology", default="full", choices=["full", "minimal"])
    p.add_argument("--scenario", default=DEFAULT_SCENARIO)
    a = p.parse_args()
    capture({"volume": a.volume, "rate": a.rate, "chunk_size": a.chunk_size,
             "jitter": a.jitter, "deadline": a.deadline}, a.window, a.seed, a.out_dir,
            topology=a.topology, scenario=a.scenario)


if __name__ == "__main__":
    main()
