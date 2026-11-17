"""Long-run C2 captures: 10-minute runs of the full training scenario, for detectors that aggregate over
more than one 60 s window and for a RITA-style beacon scorer (evasion/long_eval.py scores them offline).

Each run keeps the network up for --window seconds (evasion/capture_helper.py --long): the controlled
beacon (h2) and the periodic hosts (h3 scenario beacon, h4-h6 periodic benign) run continuously; the
bursty hosts (h7 app sessions, h8 bulk transfer) restart every 60 s with fresh parameters. Instances
are the 9 held-out instances of the C2 sweeps (3 intervals x 3 seeds, same split as evasion/harden.py),
each at every --settings (timing jitter : size jitter) pair. The background of an instance is drawn from
its seed, so every setting of an instance sees the same background parameters.

The pcap of a run is too large to keep, so after each capture this driver stores what the offline
analysis needs and deletes the pcap. Per flow (protocol, sender MAC, destination; capture host
excluded): the 17 features on each 60 s window aligned to the start of the run, the connection events
RITA would see (each UDP datagram, each TCP connection by source port, with source-side IP bytes) and,
for UDP flows, every packet's time, length and ports. Also the beacon's check-in log and its functional
verdict over the whole run. Run as root; resumes by skipping runs whose output exists:

    sudo .venv/bin/python -m evasion.long_run --out-dir results/long          # 9 x 5 runs, ~8.5 h
    sudo .venv/bin/python -m evasion.long_run --limit 1 --settings 0.3:0 --window 120 \\
         --out-dir /tmp/long_smoke                                                # smoke test, ~3 min
"""
import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from evasion import oracle
from evasion.configs import build_configs, expand_instances
from evasion.harden import split_instances
from evasion.query import SYSTEM_PYTHON
from features.dataset import FEATURE_NAMES
from features.extract import _flow_features, parse_pcap

DEFAULT_SETTINGS = "0:0,0.3:0,0.8:0,0.5:0.3,0.8:0.6"
ETH_HEADER = 14
FLAG_BITS = {"S": 1, "A": 2, "F": 4, "R": 8}


def heldout_instances(intervals, seeds_per_config, train_frac=0.67):
    configs = build_configs({"payload_size": 64}, interval=intervals)
    instances, labels = expand_instances(configs, seeds_per_config)
    _, keep = split_instances({"features": np.zeros((len(instances), 1, 1)),
                               "inst_config_idx": np.array(labels["inst_config_idx"]),
                               "inst_seed": np.array(labels["inst_seed"])}, train_frac)
    return [instances[i] for i in keep]


def run_name(seed, jitter, size_jitter):
    return f"inst{seed:02d}_j{jitter:g}_s{size_jitter:g}"


def extract_run(pcap, meta, checkins, max_interval):
    """Reduce one long-run pcap to the arrays evasion/long_eval.py needs. Returns a dict for np.savez."""
    t0, window, step = meta["window_start"], meta["window"], meta.get("restart_every", 60.0)
    n_win = int(math.ceil(window / step - 1e-9))
    roles = {h["mac"].lower(): (name, h["role"]) for name, h in meta["hosts"].items()}
    capture_mac = meta["capture_mac"].lower()
    flows = parse_pcap(pcap)

    keys = [k for k in flows if k[1] != capture_mac and k[1] in roles]
    keys.sort(key=lambda k: (roles[k[1]][0], k[0], k[2]))
    win_feat = np.full((len(keys), n_win, len(FEATURE_NAMES)), np.nan)
    win_npk = np.zeros((len(keys), n_win), int)
    conn_flow, conn_ts, conn_bytes = [], [], []
    pkt_flow, pkt_ts, pkt_len, pkt_sport, pkt_dport = [], [], [], [], []
    for fi, key in enumerate(keys):
        pk = sorted(flows[key], key=lambda p: p["ts"])
        ts = np.array([p["ts"] for p in pk])
        widx = np.floor((ts - t0) / step).astype(int)
        for w in range(n_win):
            sel = [p for p, i in zip(pk, widx) if i == w]
            win_npk[fi, w] = len(sel)
            if sel:
                row = _flow_features(key, sel)
                win_feat[fi, w] = [row[n] for n in FEATURE_NAMES]
        if key[0] == "udp":
            for p in pk:
                conn_flow.append(fi), conn_ts.append(p["ts"]), conn_bytes.append(p["length"] - ETH_HEADER)
                pkt_flow.append(fi), pkt_ts.append(p["ts"]), pkt_len.append(p["length"])
                pkt_sport.append(p["sport"] if p["sport"] is not None else -1)
                pkt_dport.append(p["dport"] if p["dport"] is not None else -1)
        elif key[0] == "tcp":
            conns = {}
            for p in pk:                         # one TCP connection per client source port
                c = conns.setdefault(p["sport"], [p["ts"], 0])
                c[1] += p["length"] - ETH_HEADER
            for first, nbytes in conns.values():
                conn_flow.append(fi), conn_ts.append(first), conn_bytes.append(nbytes)

    func = oracle.evaluate(checkins, max_interval, window_start=t0, window_end=t0 + window)
    return {
        "meta": json.dumps(meta),
        "feature_names": np.array(FEATURE_NAMES),
        "flow_proto": np.array([k[0] for k in keys]), "flow_mac": np.array([k[1] for k in keys]),
        "flow_dst": np.array([k[2] for k in keys]),
        "flow_host": np.array([roles[k[1]][0] for k in keys]),
        "flow_role": np.array([roles[k[1]][1] for k in keys]),
        "win_features": win_feat, "win_npk": win_npk,
        "conn_flow": np.array(conn_flow, int), "conn_ts": np.array(conn_ts, float),
        "conn_bytes": np.array(conn_bytes, float),
        "pkt_flow": np.array(pkt_flow, int), "pkt_ts": np.array(pkt_ts, float),
        "pkt_len": np.array(pkt_len, float), "pkt_sport": np.array(pkt_sport, int),
        "pkt_dport": np.array(pkt_dport, int),
        "checkins": np.array(checkins, float), "functional": func.functional, "max_gap": func.max_gap,
        "window_start": t0, "window": window, "step": step,
    }


def capture_one(theta, seed, window, restart_every, work):
    if work.exists():
        shutil.rmtree(work)
    cmd = [SYSTEM_PYTHON, "-m", "evasion.capture_helper", "--long",
           "--interval", str(theta["interval"]), "--jitter", str(theta["jitter"]),
           "--size-jitter", str(theta["size_jitter"]), "--payload-size", str(int(theta["payload_size"])),
           "--window", str(window), "--restart-every", str(restart_every), "--seed", str(seed),
           "--out-dir", str(work), "--topology", "full"]
    subprocess.run(cmd, check=True)
    return work / "query.pcap", json.loads((work / "meta.json").read_text()), \
        oracle.read_checkins(work / "checkins.log")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", default=DEFAULT_SETTINGS, help="comma-separated jitter:size_jitter pairs")
    p.add_argument("--intervals", default="2,4,8")
    p.add_argument("--seeds-per-config", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="capture only the first N held-out instances")
    p.add_argument("--window", type=float, default=600.0)
    p.add_argument("--restart-every", type=float, default=60.0)
    p.add_argument("--max-interval", type=float, default=30.0)
    p.add_argument("--out-dir", default="results/long")
    p.add_argument("--work-dir", default="/tmp/long_work")
    args = p.parse_args()

    settings = [tuple(float(v) for v in s.split(":")) for s in args.settings.split(",")]
    instances = heldout_instances([float(x) for x in args.intervals.split(",")], args.seeds_per_config)
    if args.limit:
        instances = instances[:args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir)
    todo = [(ci, cfg, seed, j, s) for ci, cfg, seed in instances for j, s in settings]
    print(f"{len(instances)} instances x {len(settings)} settings = {len(todo)} runs of {args.window:g} s")
    for n, (ci, cfg, seed, j, s) in enumerate(todo, 1):
        dest = out / f"{run_name(seed, j, s)}.npz"
        if dest.exists():
            print(f"[{n}/{len(todo)}] {dest.name}: done, skipping")
            continue
        theta = {**cfg, "jitter": j, "size_jitter": s}
        t = time.time()
        try:
            pcap, meta, checkins = capture_one(theta, seed, args.window, args.restart_every, work)
            rec = extract_run(pcap, meta, checkins, args.max_interval)
        except Exception as e:  # noqa: BLE001 - one failed run should not end an overnight campaign
            print(f"[{n}/{len(todo)}] {dest.name}: FAILED ({e})")
            subprocess.run(["mn", "-c"], capture_output=True)
            continue
        tmp = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **rec)
        tmp.replace(dest)
        shutil.rmtree(work, ignore_errors=True)
        print(f"[{n}/{len(todo)}] {dest.name}: {len(rec['flow_host'])} flows, "
              f"{len(rec['checkins'])} check-ins, functional={rec['functional']}, {time.time() - t:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
