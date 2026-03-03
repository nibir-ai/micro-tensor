import time
from functools import wraps

class Profiler:
    # Storage for accumulated time: { "ClassName.Method": TotalTime }
    stats = {}

    @classmethod
    def profile(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            
            # Identify which class called the method
            name = f"{args[0].__class__.__name__}.{func.__name__}"
            cls.stats[name] = cls.stats.get(name, 0) + (end - start)
            return result
        return wrapper

    @classmethod
    def report(cls):
        print("\n" + "="*60)
        print(f"{'MICRO-TENSOR PERFORMANCE AUDIT':^60}")
        print("="*60)
        print(f"{'Module.Method':<35} | {'Total Time (s)':<15}")
        print("-" * 60)
        
        # Sort by duration descending to find the biggest bottleneck
        sorted_stats = sorted(cls.stats.items(), key=lambda x: x[1], reverse=True)
        for name, duration in sorted_stats:
            print(f"{name:<35} | {duration:<15.6f}")
        print("="*60)
