from collections import deque
from collections.abc import Callable
import logging
import threading
import time
from typing import Any

from flax import nnx, struct
import jax
import optax
import psutil

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


class HardwareMonitor:
    """Samples CPU, memory, and optionally network usage in a background thread."""

    def __init__(self, n_cpus: int, interval_s: float = 15.0, monitor_network: bool = False):
        self._interval = interval_s
        self._n_cpus = n_cpus
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._max_mem_gb = 0.0

        # Target ~2 minutes of smoothing; at the 15s default this is 8 samples.
        n_window = max(1, round(120 / interval_s))
        self._cpu_window: deque[float] = deque(maxlen=n_window)
        self._mem_window: deque[float] = deque(maxlen=n_window)

        # Network monitoring setup.
        self._monitor_network = monitor_network
        if monitor_network:
            # Pick the fastest active non-loopback interface with a known link speed.
            candidates = {
                iface: stats.speed
                for iface, stats in psutil.net_if_stats().items()
                if stats.isup and stats.speed > 0 and iface != "lo"
            }
            best_iface = max(candidates, key=candidates.__getitem__, default=None)
            self.net_iface = best_iface
            self._net_capacity_bps = candidates[best_iface] * 1e6 / 8 if best_iface else 0.0

            # Snapshot initial byte counters to establish a baseline for the first interval.
            counters = psutil.net_io_counters(pernic=True)
            self._net_bytes_prev = (
                counters[best_iface].bytes_sent + counters[best_iface].bytes_recv
                if best_iface and best_iface in counters
                else 0
            )
            self._net_time_prev = time.monotonic()
            self._net_window: deque[float] = deque(maxlen=n_window)
            speed_mbps = candidates[best_iface] if best_iface else 0
            logging.info(f"Network monitoring: interface={best_iface}, link_speed={speed_mbps} Mbps")

        # Cache process objects so cpu_percent() has a baseline on subsequent calls.
        self._procs: dict[int, psutil.Process] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _get_procs(self) -> list[psutil.Process]:
        main = psutil.Process()
        current_pids = {main.pid}.union(c.pid for c in main.children(recursive=True))

        # Add new processes.
        for pid in current_pids.difference(self._procs.keys()):
            self._procs[pid] = psutil.Process(pid)

        # Remove dead processes.
        for pid in set(self._procs.keys()).difference(current_pids):
            del self._procs[pid]

        return list(self._procs.values())

    def _sample(self) -> tuple[float, float]:
        cpu_pct = 0.0
        mem_gb = 0.0
        for proc in self._get_procs():
            try:
                with proc.oneshot():
                    cpu_pct += proc.cpu_percent()
                    mem_gb += proc.memory_info().rss / 1e9
            except psutil.NoSuchProcess:
                pass
        return cpu_pct, mem_gb

    def _sample_network(self) -> float:
        """Returns current network utilization as a fraction of the fastest interface's link capacity."""
        now = time.monotonic()
        elapsed = now - self._net_time_prev

        # Skip if timing is degenerate, no capacity was detected, or no interface was found.
        if elapsed <= 0 or self._net_capacity_bps <= 0 or self.net_iface is None:
            return 0.0

        # Interface may disappear (e.g. link down); skip rather than crash.
        counters = psutil.net_io_counters(pernic=True)
        if self.net_iface not in counters:
            return 0.0

        # Compute bytes transferred since last sample and convert to a utilization percentage.
        c = counters[self.net_iface]
        bytes_now = c.bytes_sent + c.bytes_recv
        throughput_bps = (bytes_now - self._net_bytes_prev) / elapsed
        
        self._net_bytes_prev = bytes_now
        self._net_time_prev = now
        return throughput_bps / self._net_capacity_bps * 100.0

    def _run(self):
        while not self._stop.wait(self._interval):
            cpu_pct, mem_gb = self._sample()
            with self._lock:
                self._max_mem_gb = max(self._max_mem_gb, mem_gb)
                self._cpu_window.append(cpu_pct / self._n_cpus)
                self._mem_window.append(mem_gb)
                if self._monitor_network:
                    self._net_window.append(self._sample_network())

    def get(self) -> dict[str, float]:
        with self._lock:
            avg_cpu_pct = sum(self._cpu_window) / len(self._cpu_window) if len(self._cpu_window) > 0 else 0.0
            avg_mem_gb = sum(self._mem_window) / len(self._mem_window) if len(self._mem_window) > 0 else 0.0

            metrics = {
                "hardware/max_mem_rss_gb": self._max_mem_gb,
                "hardware/avg_mem_rss_gb": avg_mem_gb,
                "hardware/avg_cpu_pct": avg_cpu_pct,
            }
            if self._monitor_network:
                avg_net_pct = sum(self._net_window) / len(self._net_window) if len(self._net_window) > 0 else 0.0
                metrics["hardware/avg_net_pct"] = avg_net_pct

            return metrics

    def stop(self):
        self._stop.set()
        self._thread.join()