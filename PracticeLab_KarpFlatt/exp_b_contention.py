"""Experiment B: one shared integer, locked on every increment."""
import time
from multiprocessing import Process, Value, Lock, freeze_support
from benchmark import benchmark


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:
            shared_counter.value += 1


def run_contended(p, total_increments=2_000_000):
    if p < 1 or total_increments < 1:
        raise ValueError('All parameters must be positive')
    base, remainder = divmod(total_increments, p)
    counter = Value('i', 0)
    lock = Lock()
    procs = [Process(target=contended_worker,
                     args=(counter, lock, base + (i < remainder))) for i in range(p)]
    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - t0
    assert all(proc.exitcode == 0 for proc in procs), 'Worker process failed'
    assert counter.value == total_increments, (counter.value, total_increments)
    return elapsed


if __name__ == '__main__':
    freeze_support()
    benchmark(run_contended, 'B: Lock Contention')
