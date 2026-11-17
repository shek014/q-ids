"""Capture-only half of the C2 evasion query, run under SYSTEM python3 (which has Mininet but no
numpy). Runs ONE controlled beacon with the given parameters against a UDP echo C2, captures the
traffic, and writes the pcap + a meta sidecar + the check-in log. It does NO feature extraction (that
needs numpy/scapy and happens in the .venv half, evasion/query.py), so this file imports nothing
beyond Mininet + the pure-stdlib capture wrapper.

Topology (--topology):
  full    (default) the training scenario's own network (configs/scenarios/c2.yaml): same hosts,
          servers and background traffic as the dataset captures, so the controlled beacon shares the
          link and the contested boundary exactly as it did in training and the query flow is
          in-distribution. Background parameters are sampled once per seed, so every jitter level of an
          instance sees the same background. The controlled beacon runs on h2 against the scenario's
          echo server (C2_PORT); the scenario's OWN second beacon (h3) and every benign host run as
          background and are excluded from scoring by MAC.
  minimal the original 2-host topology (no contention, no confusers). Kept so older sweeps stay
          reproducible; its beacon flows are more regular than any training beacon because nothing else
          shares the link (out of distribution, in the MORE-detectable direction).

Long-run mode (--long, full topology only) keeps the network up for the whole --window (e.g. 600 s)
instead of one 60 s scenario run. Periodic generators (the controlled beacon h2, the scenario's own
beacon h3 and the periodic benign hosts h4-h6) run continuously with the parameters sampled once, as a
real NTP client or health check would. The bursty hosts (h7 app sessions, h8 bulk transfer) are
restarted every --restart-every seconds (default 60) with freshly sampled parameters, so each 60 s
window aligned to the start holds the same mix of traffic as one training run. meta.json then records
every host's MAC, role and sampled parameters. Driven by evasion/long_run.py.

Mirrors evasion/exfil_capture_helper.py (the exfil version). Invoked as a subprocess by
evasion/query.py; not meant to be run by hand.
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

C2_PORT = 9999                                   # matches the udp_echo server in configs/scenarios/c2.yaml
DEFAULT_SCENARIO = "configs/scenarios/c2.yaml"


def _start_full(scenario_path, seed, long_window=None):
    """Build the training scenario's network and start every server; resolve every background flow
    EXCEPT the controlled beacon (h2). Returns (net, h1, h2, servers, background, rng) where background
    is a list of dicts (host, name, module, flow, role, periodic, args, cmd). With long_window set, the
    periodic generators run for that long instead of the scenario's 60 s."""
    import random
    import yaml
    from topology.run import _resolve_args, _cmd_for, _hosts_of
    sc = yaml.safe_load(open(scenario_path))
    rng = random.Random(f"c2-query-bg-{seed}")
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
    background = []
    for role in ("benign", "attack"):
        for flow in sc.get(role, []):
            for hn in _hosts_of(flow):
                if hn == "h2":
                    continue                      # h2 is the controlled beacon
                host = net.get(hn)
                periodic = flow["module"] == "traffic.periodic_client"
                args = dict(flow.get("args", {}))
                if long_window and periodic:
                    args["duration"] = long_window    # a fixed value, so the rng draws stay the same
                resolved = _resolve_args(args, net, rng, self_host=host)
                background.append({"host": host, "name": hn, "module": flow["module"], "flow": flow,
                                   "role": flow.get("class", "benign"), "periodic": periodic,
                                   "args": resolved, "cmd": _cmd_for(flow["module"], resolved)})
    return net, h1, h2, servers, background, rng


def capture(theta, window, seed, out_dir, topology="full", scenario=DEFAULT_SCENARIO, long=False,
            restart_every=60.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcap_path = out / "query.pcap"
    log_path = out / "checkins.log"
    if log_path.exists():
        log_path.unlink()

    if long and topology != "full":
        raise SystemExit("--long needs --topology full")
    rng = None
    if topology == "full":
        net, h1, h2, servers, background, rng = _start_full(scenario, seed, window if long else None)
    else:
        net = Mininet(topo=IDSTopo(n_hosts=2), controller=OVSController, link=TCLink)
        net.start()
        h1, h2 = net.get("h1"), net.get("h2")
        for h in (h1, h2):
            for intf in h.intfList():
                if intf.name != "lo":
                    h.cmd(f"ethtool -K {intf.name} gso off tso off gro off lro off")
        net.staticArp()
        servers = [h1.popen(["python3", "-m", "traffic.udp_echo", "--ports", str(C2_PORT)])]
        background = []
    bg_macs = [b["host"].MAC() for b in background]

    bg_procs = []
    try:
        time.sleep(1.0)                                  # let servers bind

        beacon_cmd = [
            "python3", "-m", "traffic.periodic_client",
            "--target", h1.IP(), "--port", str(C2_PORT),
            "--interval", str(theta["interval"]),
            "--jitter", str(theta["jitter"]),
            "--payload-size", str(int(theta.get("payload_size", 64))),
            "--size-jitter", str(theta.get("size_jitter", 0.0)),
            "--duration", str(window),
            "--seed", str(seed),
            "--log", str(log_path),
        ]
        epochs = []
        with Capture(iface="h1-eth0", out_path=str(pcap_path), node=h1):
            if not long:
                # background starts with the beacon, as in a training run (all generators launch together)
                bg_procs = [b["host"].popen(b["cmd"]) for b in background]
                t0 = time.time()
                beacon = h2.popen(beacon_cmd)
                time.sleep(window + 1.0)
            else:
                bg_procs = [b["host"].popen(b["cmd"]) for b in background if b["periodic"]]
                t0 = time.time()
                beacon = h2.popen(beacon_cmd)
                bursty = [b for b in background if not b["periodic"]]
                k = 0
                while k * restart_every < window:
                    # restart the bursty hosts with fresh parameters at every epoch boundary
                    cur, ep = [], {"start": time.time() - t0, "hosts": {}}
                    for b in bursty:
                        args = _resolve_args(b["flow"].get("args", {}), net, rng, self_host=b["host"])
                        cur.append(b["host"].popen(_cmd_for(b["module"], args)))
                        ep["hosts"][b["name"]] = args
                    epochs.append(ep)
                    k += 1
                    time.sleep(max(0.0, t0 + min(k * restart_every, window) - time.time()))
                    for bp in cur:
                        if bp.poll() is None:
                            bp.terminate()
                time.sleep(1.0)
            if beacon.poll() is None:
                beacon.terminate()
            for bp in bg_procs:
                if bp.poll() is None:
                    bp.terminate()
        for sv in servers:
            if sv.poll() is None:
                sv.terminate()

        meta = {"beacon_mac": h2.MAC(), "capture_mac": h1.MAC(), "background_macs": bg_macs,
                "topology": topology,
                "window_start": t0, "window": window, "theta": theta, "seed": seed}
        if long:
            meta.update(long=True, restart_every=restart_every, epochs=epochs,
                        hosts={"h2": {"mac": h2.MAC(), "role": "beacon", "module": "traffic.periodic_client",
                                      "args": theta},
                               **{b["name"]: {"mac": b["host"].MAC(), "role": b["role"], "module": b["module"],
                                              "args": b["args"] if b["periodic"] else None}
                                  for b in background}})
        (out / "meta.json").write_text(json.dumps(meta))
    finally:
        for bp in bg_procs:
            if bp.poll() is None:
                bp.terminate()
        net.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=float, required=True)
    p.add_argument("--jitter", type=float, required=True)
    p.add_argument("--size-jitter", type=float, default=0.0)
    p.add_argument("--payload-size", type=int, default=64)
    p.add_argument("--window", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--topology", default="full", choices=["full", "minimal"])
    p.add_argument("--scenario", default=DEFAULT_SCENARIO)
    p.add_argument("--long", action="store_true",
                   help="long run: periodic hosts run the whole window, bursty hosts restart every --restart-every s")
    p.add_argument("--restart-every", type=float, default=60.0)
    a = p.parse_args()
    capture({"interval": a.interval, "jitter": a.jitter, "size_jitter": a.size_jitter,
             "payload_size": a.payload_size}, a.window, a.seed, a.out_dir,
            topology=a.topology, scenario=a.scenario, long=a.long, restart_every=a.restart_every)


if __name__ == "__main__":
    main()
