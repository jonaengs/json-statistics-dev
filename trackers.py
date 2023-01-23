import psutil
import tracemalloc
import time
from collections import defaultdict
import inspect

import atexit

from settings import settings
from utils import get_time_formatter
import logger

log = lambda *args, **kwargs: logger.log(*args, **kwargs, quiet=not settings.tracking.print_tracking)


class GlobalMemoryTracker:
    def __init__(self) -> None:
        self.records = defaultdict(list)
        atexit.register(self._print_results)  # ensure results are always printed

    def _print_results(self):
        log("\n" + "="*20 + "\n" + "GLOBAL MEMORY RESULTS:\n" + "="*20)
        for (file_name, file_line), mem in self.records.items():
            log(f"{file_name}:{file_line}  ({len(mem)} pass{'es'*(len(mem)>1)}):")
            
            mem = [s / (1024**2) for s in mem]  # Map from Bytes to Mega Bytes
            m_mean, m_min, m_max = sum(mem)/len(mem), min(mem), max(mem)
            log(f"PEAK: mean={m_mean:.1f}MB, min={m_min:.1f}MB, max={m_max:.1f}MB")
            
            log("-"*30)
    
    def record_global_memory(self, ):
        if not settings.tracking.global_memory:
            return

        info = psutil.Process().memory_info()

        caller_frame = inspect.currentframe().f_back
        file_path, file_line, *_ = inspect.getframeinfo(caller_frame)
        file_name = file_path.split("\\")[-1]
        
        self.records[(file_name, file_line)].append(info.rss)
        

class LocalMemoryTracker:
    """
    Uses tracemalloc to record the amount of memory allocated by a function.
    As the name implies, this tracker only checks how much memory is used locally, 
    meaning how much memory is being allocated after calling this function, not 
    how much has already been allocated. 
    """
    def __init__(self) -> None:
        self.records = defaultdict(list)
        atexit.register(self._print_results)  # ensure results are always printed

    def _print_results(self):
        log("\n" + "="*20 + "\n" + "LOCAL MEMORY RESULTS:\n" + "="*20)
        for f_name, mem in self.records.items():
            log(f"{f_name}  ({len(mem)} call{'s'*(len(mem)>1)}):")
            
            mem = [s / (1024**2) for s in mem]  # Map from Bytes to Mega Bytes
            m_mean, m_min, m_max = sum(mem)/len(mem), min(mem), max(mem)
            log(f"PEAK: mean={m_mean:.1f}MB, min={m_min:.1f}MB, max={m_max:.1f}MB")
            
            log("-"*30)

    def record_peak_memory(self, func):
        if not settings.tracking.local_memory:
            return func

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


class TimeTracker:
    def __init__(self):
        self.records = defaultdict(list)
        atexit.register(self._print_results)  # ensure results are always printed

    def _print_results(self):
        log("\n" + "="*20 + "\n" + "TIMER RESULTS:\n" + "="*20)
        for f_name, all_ts in self.records.items():
            log(f"{f_name}  ({len(all_ts)} call{'s'*(len(all_ts)>1)}):")
            
            sys_ts = list(zip(*all_ts))[0]
            s_mean, s_min, s_max = sum(sys_ts)/len(sys_ts), min(sys_ts), max(sys_ts)
            formatter = get_time_formatter(s_mean)
            log(f"SYSTEM: mean={formatter(s_mean)} min={formatter(s_min)} max={formatter(s_max)} total={formatter(sum(sys_ts))}")
            
            prog_ts = list(zip(*all_ts))[1]
            p_mean, p_min, p_max = sum(prog_ts)/len(prog_ts), min(prog_ts), max(prog_ts)
            log(f"PROGRAM: mean={formatter(p_mean)} min={formatter(p_min)} max={formatter(p_max)} total={formatter(sum(prog_ts))}")

            log("-"*30)


    def record_time_used(self, func):
        if not settings.tracking.time:
            return func

        def wrapper(*args, **kwargs):
            t0_system = time.process_time_ns()
            t0_program = time.perf_counter_ns()

            result = func(*args, **kwargs)
            
            t1_system = time.process_time_ns()
            t1_program = time.perf_counter_ns()
        
            # Divide by 1M to get ms time 
            delta_system = t1_system - t0_system
            delta_program = t1_program - t0_program

            self.records[func.__name__].append((delta_system, delta_program))

            return result

        wrapper.__name__ = func.__name__  # Avoid nested decorators losing name of decorated function
        return wrapper


global_mem_tracker = GlobalMemoryTracker()
local_mem_tracker = LocalMemoryTracker()
time_tracker = TimeTracker()
