"""
System-level telemetry probes (CPU, RAM, disk, network).
Uses psutil; degrades gracefully when metrics are unavailable.
"""

import os
import time
import psutil
from typing import Dict, Optional


# Cache the current process handle
_PROCESS = psutil.Process(os.getpid())


def probe_process() -> Dict[str, float]:
    """
    Process-level metrics: RSS, VMS, CPU%, threads, open files.
    All values are best-effort; missing metrics return 0.0.
    """
    out = {
        'rss_mb': 0.0,
        'vms_mb': 0.0,
        'cpu_percent': 0.0,
        'num_threads': 0,
        'num_open_files': 0,
    }
    try:
        mem = _PROCESS.memory_info()
        out['rss_mb'] = mem.rss / (1024.0 * 1024.0)
        out['vms_mb'] = mem.vms / (1024.0 * 1024.0)
        out['cpu_percent'] = _PROCESS.cpu_percent(interval=None)
        out['num_threads'] = _PROCESS.num_threads()
        try:
            out['num_open_files'] = _PROCESS.num_fds() if hasattr(_PROCESS, 'num_fds') else len(_PROCESS.open_files())
        except Exception:
            out['num_open_files'] = 0
    except Exception:
        pass
    return out


def probe_system() -> Dict[str, float]:
    """
    System-level metrics: global CPU%, memory, swap, disk, uptime.
    """
    out = {
        'cpu_percent': 0.0,
        'mem_percent': 0.0,
        'mem_available_mb': 0.0,
        'mem_total_mb': 0.0,
        'swap_percent': 0.0,
        'disk_percent': 0.0,
        'uptime_s': 0.0,
        'cpu_count': 1,
    }
    try:
        out['cpu_percent'] = psutil.cpu_percent(interval=None)
        out['cpu_count'] = psutil.cpu_count(logical=True) or 1

        vm = psutil.virtual_memory()
        out['mem_percent'] = vm.percent
        out['mem_available_mb'] = vm.available / (1024.0 * 1024.0)
        out['mem_total_mb'] = vm.total / (1024.0 * 1024.0)

        try:
            sm = psutil.swap_memory()
            out['swap_percent'] = sm.percent
        except Exception:
            pass

        try:
            out['disk_percent'] = psutil.disk_usage('/').percent
        except Exception:
            pass

        try:
            out['uptime_s'] = time.time() - psutil.boot_time()
        except Exception:
            pass
    except Exception:
        pass
    return out


def snapshot_all() -> Dict[str, Dict[str, float]]:
    """Return both process and system telemetry in a single dict."""
    return {
        'process': probe_process(),
        'system': probe_system(),
    }