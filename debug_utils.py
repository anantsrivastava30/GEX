import concurrent.futures
import threading
import time
import psutil
import logging
import os

class DebugThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """ThreadPoolExecutor that logs task runtime and resource usage."""

    def __init__(self, *args, label="", **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.proc = psutil.Process(os.getpid())

    def _wrap(self, fn, *args, **kwargs):
        tid = threading.get_ident()
        start_cpu = self.proc.cpu_times()
        start_mem = self.proc.memory_info().rss
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        duration = time.perf_counter() - start
        end_cpu = self.proc.cpu_times()
        end_mem = self.proc.memory_info().rss
        cpu_time = (end_cpu.user + end_cpu.system) - (
            start_cpu.user + start_cpu.system
        )
        mem_diff = end_mem - start_mem
        logging.debug(
            f"[{self.label}] thread {tid} ran {fn.__name__} in {duration:.2f}s;"
            f" CPU {cpu_time:.2f}s; Δmem {mem_diff/1e6:.2f}MB;"
            f" total threads {self.proc.num_threads()}"
        )
        return result

    def submit(self, fn, *args, **kwargs):
        return super().submit(self._wrap, fn, *args, **kwargs)

def get_executor(max_workers, label=""):
    """Return debug executor if DEBUG_THREADS env var set, else normal."""
    if os.getenv("DEBUG_THREADS") == "1":
        logging.basicConfig(level=logging.DEBUG)
        return DebugThreadPoolExecutor(max_workers=max_workers, label=label)
    return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
