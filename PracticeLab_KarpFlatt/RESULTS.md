# Empirical Scalability Diagnostics: Karp-Flatt Metric

Student: Assimbek Sultan
Student ID: 230103379
Branch: `practice-230103379`

Location: `PracticeLab_KarpFlatt/` (separate folder requested by the student; report is at the lab folder root).
Measurement date: 2026-10-29 (Asia/Almaty)

## Hardware and method

- Host: macOS 15.6, ARM64; 10 logical CPUs reported by `os.cpu_count()`.
- Physical core count, exact chip model and RAM: unavailable because host sysctl access was denied; no values inferred.
- Python: 3.12.14; multiprocessing start method: spawn.
- Standard library only; three repetitions per worker count, interleaved P=1,2,4; medians reported. Experiments run sequentially.
- Timings include process startup and teardown; A additionally includes the prescribed 0.40 s delay. No warm pool or subtracted overhead.
- Workloads unchanged from the handout: A = 16 chunks of 2,000,000; B = 2,000,000 increments; C = 20,000,000 increments.
- Raw individual measurements are retained in `measurements/`.

## Formulas

`S(p) = T1 / Tp`
`e(p) = (p * Tp / T1 - 1) / (p - 1)` for p >= 2
`E(p) = S(p) / p`
Karp-Flatt is undefined at p=1. Values above 1 reflect overhead and are not literal serial code fractions.

## Required results

| Experiment | T1 (s) | T2 (s) | T4 (s) | S2 | S4 | e(2) | e(4) | E4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Serial Tail | 2.189773 | 1.355551 | 0.989383 | 1.615412 | 2.213272 | 0.238075 | 0.269093 | 55.33% |
| B: Lock Contention | 3.437021 | 9.247292 | 8.446912 | 0.371679 | 0.406897 | 4.380992 | 2.943501 | 10.17% |
| C: Privatized State | 0.376646 | 0.202254 | 0.149960 | 1.862246 | 2.511645 | 0.073972 | 0.197527 | 62.79% |

## Interpretation and rubric audit

### A: fixed sequential tail - flatness criterion PASS

The e(4)/e(2) ratio is 1.1303, inside [0.80, 1.25]. The fixed delay limits speedup. The absolute e values exceed the example range 0.10-0.14: the measured T1 is only 2.190 s, so 0.40 s alone is about 18.3% of T1; process overhead also contributes. The example serial fraction is not portable across hardware.

### B: severe contention - slowdown PASS; increasing-e criterion NOT MET

T4 > T1 (8.447 s > 3.437 s), and both parallel speedups are below one. However, e(4) < e(2), contrary to the suggested monotonic signature. This is the actual measurement, not an arithmetic error. Karp-Flatt measures effective serialization plus overhead: it does not mathematically identify a unique cause. Here the code deliberately locks a shared counter on every increment, supporting lock contention as the mechanism. Cache-line bouncing and kernel waits were not separately profiled; their individual contribution cannot be claimed from timings alone. With Value('i', 0), the internal synchronized-value lock also remains as prescribed.

### C: privatization - speedup PASS; default efficiency criterion NOT MET

Each process counts privately, and the parent sums results after map completes. T4 < T1, with S4=2.512 and E4=62.79%. This does not meet E4 > 70% or the suggested e <= 0.03 at the required default workload. Spawn startup, IPC and final reduction are included in the short runtime. Privatization removes the per-increment shared lock, but does not eliminate multiprocessing overhead. B and C use different operation counts, so their absolute runtimes are not a same-size comparison.

## Supplementary C: amortizing startup overhead

This additional run uses 200,000,000 increments (10 times the default). It is explicitly supplementary; it does not replace the required 20,000,000-increment measurements.

| T1 (s) | T2 (s) | T4 (s) | S4 | E4 | e(2) | e(4) |
|---:|---:|---:|---:|---:|---:|---:|
| 3.114472 | 1.612875 | 0.930271 | 3.347919 | 83.70% | 0.035729 | 0.064924 |

The larger workload achieves E4 > 70%, consistent with fixed startup costs becoming less significant. Even here e(4) is above 0.03; ideal scaling is not assumed.

## Reproduction

Run from `PracticeLab_KarpFlatt/` with Python 3.8+; no packages required.

```sh
python3 -c "import os; print(f'Execution Cores: {os.cpu_count()}')"
python3 verify_small.py
python3 run_all.py
python3 check_large_c.py --repeats 3 --json measurements/c_large.json
```

Individual runs:

```sh
python3 exp_a_serial.py --repeats 3 --json measurements/a.json
python3 exp_b_contention.py --repeats 3 --json measurements/b.json
python3 exp_c_fixed.py --repeats 3 --json measurements/c.json
```

Re-running overwrites raw JSON; update this report if new measurements are used. `run_all.py` runs the prescribed workloads only.

## Correctness verification

`python3 verify_small.py`: PASS. Verified the mathematical kernel, private accumulation, exact shared/private totals for P=1,2,4 with 101 increments (including remainders), and the sequential delay. Every full B/C benchmark also asserts the exact final count; B checks worker exit codes. Main guards and module-level worker functions support spawn on macOS/Windows.

## Conclusion

The fixed delay gives approximately flat effective serialization in A. Shared-state locking makes B slower than one process, although its e values are not monotonic. Private accumulation restores speedup in C; the required small workload misses the efficiency threshold on this host, while the separately labelled larger workload exceeds it. All reported values are measured; expected handout values were not substituted.

The handout example table contains arithmetic inconsistencies (for example B: T1=0.65 and T4=2.40 imply e(4)=4.5897, not 3.920). This report uses the stated formula.
