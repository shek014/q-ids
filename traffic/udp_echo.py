"""Minimal UDP echo server — the far end for traffic.periodic_client (C2 endpoint / benign service).
Binds one or more UDP ports and echoes every datagram back to its sender, so a check-in gets a reply
(the functional-oracle signal). Runs until terminated by the scenario runner. No root needed.
"""
import argparse
import socket
import threading


def _serve(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        return
    while True:
        try:
            data, addr = s.recvfrom(4096)
            s.sendto(data, addr)
        except OSError:
            break


def run(ports):
    threads = [threading.Thread(target=_serve, args=(p,), daemon=True) for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # blocks until the process is killed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", default="9999")
    args = parser.parse_args()
    run([int(p) for p in args.ports.split(",")])


if __name__ == "__main__":
    main()
