"""Execution safeguards for this shared server; never changes model/data policy."""
import os
import threading
import time
from pathlib import Path


def preserve_cpu_affinity():
    """Some numerical-library imports pin the main thread to one busy core.

    Preserve only the launcher-assigned CPU set, never expand its allocation.
    Spawn data workers are used so this daemon is never inherited through fork.
    """
    if os.environ.get("FLOWERS_PRESERVE_AFFINITY") != "1" or not hasattr(os, "sched_getaffinity"):
        return
    assigned = set(os.sched_getaffinity(0))
    def maintain():
        while True:
            for task in Path("/proc/self/task").iterdir():
                try:
                    if set(os.sched_getaffinity(int(task.name))) != assigned:
                        os.sched_setaffinity(int(task.name), assigned)
                except (ProcessLookupError, FileNotFoundError):
                    pass
            time.sleep(3)
    threading.Thread(target=maintain, daemon=True, name="assigned-cpu-affinity").start()


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", OMP_PROC_BIND="false",
        KMP_AFFINITY="disabled", FLOWERS_SPAWN_WORKERS="1", FLOWERS_PRESERVE_AFFINITY="1",
        FLOWERS_DATA_WORKERS="0")
