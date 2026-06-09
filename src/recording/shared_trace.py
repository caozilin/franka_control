from __future__ import annotations

import csv
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

RAW_TRACE_WIDTH = 1 + 16 + 16 + 16 + 7
RAW_TRACE_FIELDS = (
    ["time"]
    + [f"goal_pose_{i}" for i in range(16)]
    + [f"ref_pose_{i}" for i in range(16)]
    + [f"actual_pose_{i}" for i in range(16)]
    + [f"tau_cmd_j{i + 1}" for i in range(7)]
)


class SharedTraceRing:
    def __init__(self, capacity: int, width: int = RAW_TRACE_WIDTH, *, name: str | None = None, create: bool = True):
        self.capacity = int(capacity)
        self.width = int(width)
        self._data_nbytes = self.capacity * self.width * np.dtype(np.float64).itemsize
        self._seq_nbytes = self.capacity * np.dtype(np.int64).itemsize
        if create:
            self.data_shm = shared_memory.SharedMemory(create=True, size=self._data_nbytes)
            self.seq_shm = shared_memory.SharedMemory(create=True, size=self._seq_nbytes)
        else:
            if name is None:
                raise ValueError("name is required when create=False")
            data_name, seq_name = name.split(":", 1)
            self.data_shm = shared_memory.SharedMemory(name=data_name)
            self.seq_shm = shared_memory.SharedMemory(name=seq_name)

        self.data = np.ndarray((self.capacity, self.width), dtype=np.float64, buffer=self.data_shm.buf)
        self.seq = np.ndarray((self.capacity,), dtype=np.int64, buffer=self.seq_shm.buf)
        self.counter = 0
        if create:
            self.data.fill(0.0)
            self.seq.fill(-1)

    @property
    def name(self) -> str:
        return f"{self.data_shm.name}:{self.seq_shm.name}"

    def write_raw(self, elapsed: float, goal_pose, ref_pose, actual_pose, tau_cmd) -> None:
        idx = self.counter % self.capacity
        row = self.data[idx]
        row[0] = float(elapsed)
        row[1:17] = goal_pose
        row[17:33] = ref_pose
        row[33:49] = actual_pose
        row[49:56] = tau_cmd
        self.seq[idx] = self.counter
        self.counter += 1

    def close(self) -> None:
        self.data_shm.close()
        self.seq_shm.close()

    def unlink(self) -> None:
        self.data_shm.unlink()
        self.seq_shm.unlink()


def _logger_worker(shm_name: str, capacity: int, width: int, output_path: str, stop_event, poll_period: float) -> None:
    ring = SharedTraceRing(capacity, width, name=shm_name, create=False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_seq = -1
    try:
        with output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(RAW_TRACE_FIELDS)
            while not stop_event.is_set():
                last_seq = _drain_available_rows(ring, writer, last_seq)
                output_file.flush()
                time.sleep(float(poll_period))
            last_seq = _drain_available_rows(ring, writer, last_seq)
            output_file.flush()
    finally:
        ring.close()


def _drain_available_rows(ring: SharedTraceRing, writer: csv.writer, last_seq: int) -> int:
    max_seq = int(ring.seq.max())
    if max_seq <= last_seq:
        return last_seq
    start_seq = max(last_seq + 1, max_seq - ring.capacity + 1)
    for seq in range(start_seq, max_seq + 1):
        idx = seq % ring.capacity
        if int(ring.seq[idx]) == seq:
            writer.writerow(ring.data[idx].copy().tolist())
    return max_seq


class AsyncSharedTraceRecorder:
    def __init__(self, output_path: Path, *, capacity: int, width: int = RAW_TRACE_WIDTH, poll_period: float = 0.02):
        self.ring = SharedTraceRing(capacity, width, create=True)
        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=_logger_worker,
            args=(self.ring.name, int(capacity), int(width), str(output_path), self._stop_event, float(poll_period)),
            daemon=True,
        )
        self._process.start()

    def write_raw(self, elapsed: float, goal_pose, ref_pose, actual_pose, tau_cmd) -> None:
        self.ring.write_raw(elapsed, goal_pose, ref_pose, actual_pose, tau_cmd)

    def close(self) -> None:
        self._stop_event.set()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self.ring.close()
        self.ring.unlink()
