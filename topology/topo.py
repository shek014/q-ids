"""Mininet topology: N hosts on a single OVS switch. Linux + Mininet only — see
README > Prerequisites. Not importable on Windows (mininet depends on Linux network
namespaces).
"""
from mininet.topo import Topo


class IDSTopo(Topo):
    def build(self, n_hosts=4):
        switch = self.addSwitch("s1")
        for i in range(1, n_hosts + 1):
            host = self.addHost(f"h{i}")
            self.addLink(host, switch)
