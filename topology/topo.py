"""Mininet topology: N hosts on a single OVS switch. Linux + Mininet only — see
README > Prerequisites. Not importable on Windows (mininet depends on Linux network
namespaces).
"""
from mininet.topo import Topo


class IDSTopo(Topo):
    def build(self, n_hosts=4, bw=10):
        # bw caps each link at `bw` Mbit/s (needs link=TCLink, set in run.py). Without it,
        # Mininet veth links run at multi-Gbit/s and a single benign iperf flow produces
        # tens of GB of capture in seconds. 10 Mbit/s is a realistic segment and bounds volume.
        switch = self.addSwitch("s1")
        for i in range(1, n_hosts + 1):
            host = self.addHost(f"h{i}")
            self.addLink(host, switch, bw=bw)
