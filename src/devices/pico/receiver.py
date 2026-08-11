from __future__ import annotations

from dataclasses import dataclass
import socket
import threading
import time

from devices.pico.protocol import PicoPacket


@dataclass(frozen=True)
class PicoSnapshot:
    packet: PicoPacket
    received_monotonic_s: float

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else float(now)) - self.received_monotonic_s


class PicoUdpReceiver:
    """Background UDP receiver exposing only the newest controller state."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9010) -> None:
        self.host = str(host)
        self.port = int(port)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: PicoSnapshot | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind((self.host, self.port))
        udp_socket.settimeout(0.1)
        self.port = int(udp_socket.getsockname()[1])
        self._socket = udp_socket
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pico-udp", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        udp_socket = self._socket
        if udp_socket is None:
            return
        while not self._stop.is_set():
            try:
                payload, _address = udp_socket.recvfrom(65535)
                packet = PicoPacket.from_json(payload)
            except socket.timeout:
                continue
            except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError):
                continue
            snapshot = PicoSnapshot(packet, time.monotonic())
            with self._lock:
                previous = self._latest
                if (
                    previous is None
                    or packet.session_id != previous.packet.session_id
                    or packet.sequence > previous.packet.sequence
                ):
                    self._latest = snapshot

    def latest(self) -> PicoSnapshot | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        udp_socket = self._socket
        self._socket = None
        if udp_socket is not None:
            udp_socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
