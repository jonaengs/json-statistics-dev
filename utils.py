import psutil
import tracemalloc
import time
from collections import defaultdict
import inspect

import atexit

from logger import log


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

class MemoryTracker:
    def __init__(self) -> None:
        self.records = defaultdict(list)

        atexit.register(self.print_results)

    def print_results(self):
        log("\n" + "="*20 + "\n" + "MEMORY RESULTS:\n" + "="*20)
        for f_name, mem in self.records.items():
            log(f"{f_name}  ({len(mem)} call{'s'*(len(mem)>1)}):")
            
            mem = [s / (1024**2) for s in mem]  # Map from Bytes to Mega Bytes
            m_mean, m_min, m_max = sum(mem)/len(mem), min(mem), max(mem)
            log(f"PEAK: mean={m_mean:.1f}MB, min={m_min:.1f}MB, max={m_max:.1f}MB, total={sum(mem):.1f}MB")
            
            log("-"*30)

    def record_peak_memory(self, func):
        def wrapper(*args, **kwargs):
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            tracemalloc.reset_peak()

            result = func(*args, **kwargs)
            
            _size, peak = tracemalloc.get_traced_memory()
            self.records[func.__name__].append(peak)

            return result
        
        wrapper.__name__ = func.__name__  # Avoid nested decorators losing name of decorated function
        return wrapper


class Timer:
    def __init__(self):
        self.records = defaultdict(list)

        # ensure results are always printed
        atexit.register(self.print_results)

    def print_results(self):
        log("\n" + "="*20 + "\n" + "TIMER RESULTS:\n" + "="*20)
        for f_name, all_ts in self.records.items():
            log(f"{f_name}  ({len(all_ts)} call{'s'*(len(all_ts)>1)}):")
            
            sys_ts = list(zip(*all_ts))[0]
            s_mean, s_min, s_max = sum(sys_ts)/len(sys_ts), min(sys_ts), max(sys_ts)
            log(f"SYSTEM: mean={s_mean:.1f}ms, min={s_min:.1f}ms, max={s_max:.1f}ms, total={sum(sys_ts):.1f}ms")
            
            prog_ts = list(zip(*all_ts))[1]
            p_mean, p_min, p_max = sum(prog_ts)/len(prog_ts), min(prog_ts), max(prog_ts)
            log(f"PROGRAM: mean={p_mean:.1f}ms, min={p_min:.1f}ms, max={p_max:.1f}ms, total={sum(prog_ts):.1f}ms")

            log("-"*30)


    def record_time_used(self, func):
        def wrapper(*args, **kwargs):
            t0_system = time.perf_counter_ns()
            t0_program = time.process_time_ns()

            result = func(*args, **kwargs)
            
            t1_system = time.perf_counter_ns()
            t1_program = time.process_time_ns()
        
            # Divide by 1M to get ms time 
            delta_system = (t1_system - t0_system) / 1_000_000
            delta_program = (t1_program - t0_program) / 1_000_000

            self.records[func.__name__].append((delta_system, delta_program))

            return result

        wrapper.__name__ = func.__name__  # Avoid nested decorators losing name of decorated function
        return wrapper


mem_tracker = MemoryTracker()
timer = Timer()
