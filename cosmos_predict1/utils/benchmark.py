import time
import functools

import torch

from cosmos_predict1.utils import log


def benchmark_decorator(profiling_iterations_count=3, warmup_iterations_count=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            time_sum = 0
            result = None
            total_runs = warmup_iterations_count + profiling_iterations_count
            
            for i in range(total_runs):
                if i >= warmup_iterations_count:
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                
                result = func(*args, **kwargs)
                
                if i >= warmup_iterations_count:
                    torch.cuda.synchronize()
                    time_sum += time.perf_counter() - start_time
            
            if profiling_iterations_count > 0:
                log.critical(f"The benchmarked generation time for {func.__name__} is {time_sum / profiling_iterations_count} seconds.")
            
            return result
        
        return wrapper
    
    return decorator
