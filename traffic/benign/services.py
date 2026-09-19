"""Lightweight multi-port TCP services for the victim host, so benign clients have real open
ports to connect to — a realistic host runs several services, not one. Each port accepts a
connection, echoes a small reply, and closes. Runs until terminated by the scenario runner.
No root needed.
"""
import argparse
import socket
import threading


def _serve(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        s.listen(32)
    except OSError:
        return
    while True:
        try:
            conn, _ = s.accept()
        except OSError:
            break
        try:
            conn.recv(4096)
            conn.sendall(b"HTTP/1.0 200 OK\r\n\r\nok\n")
        except OSError:
            pass
        finally:
            conn.close()


def run(ports):
    threads = [threading.Thread(target=_serve, args=(p,), daemon=True) for p in ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # blocks until the process is killed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", default="80,443,8080,22,3306,5432,8000,9000,25,110")
    args = parser.parse_args()
    run([int(p) for p in args.ports.split(",")])


if __name__ == "__main__":
    main()
