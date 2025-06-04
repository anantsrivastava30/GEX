import threading
import psutil
import time
import os

_original_start = threading.Thread.start
_original_run = threading.Thread.run


def _patched_start(self, *args, **kwargs):
    self._debug_start_time = time.perf_counter()
    target = getattr(self, "_target", None)
    print(
        f"[thread-debug] starting {self.name} target={getattr(target, '__name__', target)} "
        f"active={threading.active_count()}"
    )
    return _original_start(self, *args, **kwargs)


def _patched_run(self, *args, **kwargs):
    result = _original_run(self, *args, **kwargs)
    duration = time.perf_counter() - getattr(self, "_debug_start_time", time.perf_counter())
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(
        f"[thread-debug] finished {self.name} in {duration:.3f}s mem={mem_mb:.2f}MB "
        f"active={threading.active_count()}"
    )
    return result


def enable():
    if getattr(threading, "_debug_enabled", False):
        return
    threading.Thread.start = _patched_start
    threading.Thread.run = _patched_run
    threading._debug_enabled = True
    print("[thread-debug] enabled")
