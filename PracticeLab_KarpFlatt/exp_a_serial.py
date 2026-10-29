"""Experiment A: fixed sequential delay followed by parallel mathematics."""
import math
import time
from multiprocessing import Pool, freeze_support
from benchmark import benchmark


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    if workers < 1 or total_chunks < 1 or chunk_size < 1:
        raise ValueError('All parameters must be positive')
    t0 = time.perf_counter()
    time.sleep(0.40)
    with Pool(processes=workers) as pool:
        results = pool.map(heavy_math, [chunk_size] * total_chunks)
    elapsed = time.perf_counter() - t0
    assert len(results) == total_chunks and all(math.isfinite(x) for x in results)
    return elapsed


if __name__ == '__main__':
    freeze_support()
    benchmark(run_experiment, 'A: Serial Tail')
