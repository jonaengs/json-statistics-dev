from typing import Callable
import psutil
import tracemalloc
import time
from collections import defaultdict, namedtuple
import inspect

from settings import settings
import logger

log = lambda *args, **kwargs: logger.log(*args, **kwargs, quiet=not settings.tracking.print_tracking)

def print_total_memory_use():
    info = psutil.Process().memory_info()
    # log("RSS:", info.rss//(1024**2), info.rss//1024, info.rss)
    # log("VMS:", info.vms//(1024**2), info.vms//1024, info.vms)

    caller_frame = inspect.currentframe().f_back
    file_path, file_line, *_ = inspect.getframeinfo(caller_frame)
    file_name = file_path.split("\\")[-1]
    log(f"{file_name}:{file_line} RSS used: {info.rss/(1024**2):.1f} MB")


def print_peak_memory(f):
    def wrapper(*args, **kwargs):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.reset_peak()

        result = f(*args, **kwargs)
        
        _size, peak = tracemalloc.get_traced_memory()
        log(f"{f.__name__} peak memory use: {peak/(1024**2):.1f} MB")

        return result

    return wrapper


def get_time_formatter(t) -> Callable[[int|float], str]:
    """ takes a time in ns and finds the best format (s, ms, μs, ns)"""
    i = 0
    while t >= 1000 and i < 3:
        t /= 1000
        i += 1

    sizes = ["ns", "us", "ms", "s"]
    return lambda _t : f"{_t//(1000**i):.0f}{sizes[i]}"


def print_time_used(f):
    def wrapper(*args, **kwargs):
        t0_system = time.perf_counter_ns()
        t0_program = time.process_time_ns()

        result = f(*args, **kwargs)
        
        t1_system = time.perf_counter_ns()
        t1_program = time.process_time_ns()
       
        # Divide by 1M to get ms time 
        delta_system = (t1_system - t0_system) / 1_000_000
        delta_program = (t1_program - t0_program) / 1_000_000

        log(f"{f.__name__} time spent: {delta_system:.1f} ms, {delta_program:.1f} ms")

        return result

    return wrapper
