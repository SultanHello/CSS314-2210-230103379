"""Experiment C: private accumulation followed by one final reduction."""
import time
from multiprocessing import Pool, freeze_support
from benchmark import benchmark


def private_worker(iterations):
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def run_privatized(p, total_increments=20_000_000):
    if p < 1 or total_increments < 1:
        raise ValueError('All parameters must be positive')
    base, remainder = divmod(total_increments, p)
    tasks = [base + (i < remainder) for i in range(p)]
    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)
    elapsed = time.perf_counter() - t0
    assert total == total_increments, (total, total_increments)
    return elapsed


if __name__ == '__main__':
    freeze_support()
    benchmark(run_privatized, 'C: Privatized State')
