"""TCP sink for data exfiltration — the external collector / drop server (PROTOTYPE).

The far end of traffic.exfil_client, and the exfil analogue of traffic.udp_echo in the C2 substrate:
it accepts connections, drains all bytes until the client closes its write side, and counts how many
it received per connection. With --log it records, for each completed connection, the total bytes
received and a timestamp — the collector-side ground truth for the exfil functional oracle (the full
payload actually arrived). Binds one or more TCP ports; runs until the scenario runner kills it. No
root needed.
"""
import argparse
import socket
import threading
import time


def _handle(conn, logf, log_lock):
    total = 0
    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            total += len(data)
    except OSError:
        pass
    finally:
        conn.close()
    if logf:
        with log_lock:
            logf.write(f"{time.time():.6f} {total}\n")
            logf.flush()


def _serve(port, logf, log_lock):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        s.listen(16)
    except OSError:
        return
    while True:
        try:
            conn, _ = s.accept()
        except OSError:
            break
        threading.Thread(target=_handle, args=(conn, logf, log_lock), daemon=True).start()


def run(ports, log=None):
    logf = open(log, "a") if log else None
    log_lock = threading.Lock()
    threads = [threading.Thread(target=_serve, args=(p, logf, log_lock), daemon=True) for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # blocks until the process is killed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", default="8443")
    parser.add_argument("--log", default=None, help="append '<timestamp> <bytes_received>' per connection here")
    args = parser.parse_args()
    run([int(p) for p in args.ports.split(",")], args.log)


if __name__ == "__main__":
    main()
