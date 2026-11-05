# Task 22 - Empirical weak scaling

Requires Python 3.10+, NumPy; macOS telemetry uses system_profiler/sysctl. Other systems also require psutil.
Run from the repository root:

```sh
python3 benchmarks/weak_scale_bench/benchmark.py --repeats 5
python3 benchmarks/fit_gustafson.py
```

Default counts are every integer from 1 through os.cpu_count(), including the maximum hardware concurrency. Each worker handles 960,000 float64 elements and 12 identical transformation/reduction passes. Thus total useful element-passes = P * 960000 * 12. Inputs vary deterministically by task index; serial and parallel paths use identical task lists.

Each independent trial starts a fresh spawned pool, finishes imports and warmup, and waits for all workers to become ready. The recorded parallel interval includes dispatch, a P-worker start barrier, array allocation, NumPy computation and returned scalar reductions. Pool startup/import and teardown are excluded; startup-to-ready is saved separately. A task barrier ensures the P tasks use P distinct workers. Native numerical libraries are limited to one thread per process. Serial baseline executes those P tasks sequentially in the parent (already imported/warmed). Trial order is deterministically randomized and serial-first/parallel-first order alternates across repetitions.

This is a process-based synthetic numerical benchmark, not a network or DRAM bandwidth benchmark. Numerical checks compare serial/parallel sums. The CSV contains all five runs, not just selected fast samples. It is normal for timing noise and mixed-speed cores to disturb the ideal weak-scaling curve.
