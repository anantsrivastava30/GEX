import threading
import time
import os
import psutil

_original_start = threading.Thread.start
_original_run = threading.Thread.run
_process = psutil.Process(os.getpid())


def _patched_start(self, *args, **kwargs):
    target = getattr(self, "_target", None)
    print(
        f"[thread-debug] spawn name={self.name} target={getattr(target, '__name__', target)} "
        f"active={threading.active_count() + 1}"
    )
    return _original_start(self, *args, **kwargs)


def _cpu_snapshot():
    return {t.id: (t.user_time, t.system_time) for t in _process.threads()}


def _patched_run(self, *args, **kwargs):
    tid = getattr(self, "native_id", None)
    start_cpu = _cpu_snapshot()
    start_mem = _process.memory_info().rss
    start = time.perf_counter()
    try:
        return _original_run(self, *args, **kwargs)
    finally:
        end = time.perf_counter()
        duration = end - start
        end_cpu = _cpu_snapshot()
        end_mem = _process.memory_info().rss
        cpu = 0.0
        if tid in start_cpu and tid in end_cpu:
            cpu = (end_cpu[tid][0] - start_cpu[tid][0]) + (
                end_cpu[tid][1] - start_cpu[tid][1]
            )
        mem_delta = (end_mem - start_mem) / (1024 * 1024)
        print(
            f"[thread-debug] done name={self.name} time={duration:.3f}s cpu={cpu:.3f}s mem+={mem_delta:.2f}MB "
            f"active={threading.active_count() - 1}"
        )


def enable():
    if getattr(threading, "_debug_enabled", False):
        return
    threading.Thread.start = _patched_start
    threading.Thread.run = _patched_run
    threading._debug_enabled = True
    print("[thread-debug] enabled")
