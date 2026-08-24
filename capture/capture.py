"""tcpdump orchestration — start/stop a capture around a traffic scenario. Requires root
and a Linux host with tcpdump installed (see README > Prerequisites); not runnable on Windows.
"""
import subprocess
import time


class Capture:
    """Context manager wrapping a background tcpdump process.

    with Capture(iface="h2-eth0", out_path="capture/run.pcap") as cap:
        ...run traffic...
    """

    def __init__(self, iface, out_path, bpf_filter=None, snaplen=262144):
        self.iface = iface
        self.out_path = out_path
        self.bpf_filter = bpf_filter
        self.snaplen = snaplen
        self._proc = None

    def start(self):
        cmd = ["tcpdump", "-i", self.iface, "-w", self.out_path, "-s", str(self.snaplen), "-U"]
        if self.bpf_filter:
            cmd.append(self.bpf_filter)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)  # give tcpdump a moment to attach before traffic starts
        return self

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
