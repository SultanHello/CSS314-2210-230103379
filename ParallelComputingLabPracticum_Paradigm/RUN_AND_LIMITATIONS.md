# How to Run the Lab and Current Status

Student ID: **30103379**. N = **1472**, M = **736**, seed = **65598**.

## Files and folders

- `src/`: Python scripts and one C file, `kernels.c`. Python calls the C functions through ctypes.
- `results/`: real measurements from this computer, saved output arrays, and local check results.
- `README.md`: a short summary of the findings.
- The repository has five Git tags, one for each lab stage. Git history is stored in `.git/` at the repository root.

## How to run

You need Python 3.10 or newer, NumPy, and a C compiler called `cc`. On macOS, the compiler is part of Command Line Tools.

Open a terminal in the lab folder and run:

```sh
python3 -m pip install -r requirements.txt
python3 run_all.py 30103379
```

This replaces the old result files. It does not create new Git commits or change the tags.

You can also run each stage separately:

```sh
python3 src/profile_env.py --student-id 30103379
python3 src/stride_bench.py
python3 src/dense_kernel.py
python3 src/false_sharing.py
python3 src/scaling_study.py
python3 verify_local.py
```

## How the tests work

All tests use `perf_counter_ns` to measure time.

- **Stride:** 5 warmup runs, then 5 measured runs.
- **Matrices:** 1 warmup run, then 3 measured runs. The result uses the median, which is the middle time.
- **Counters:** 3 measured runs. Each process adds 1 to its counter 5,000,000 times. Every final counter is checked.
- **Stencil:** 1 warmup run, then 80 passes over the same input. The result uses the median of 3 measured runs.

Processes share memory through RawArray. The counters do not use one common lock. Counter and stencil times do not include process startup. Timing starts when the first worker starts its measured loop and ends when the last worker finishes. Delays from the operating system can still affect the results. Workers are not assigned to specific CPU cores.

Python controls the tests and saves the results. The loops run in `kernels.c` to reduce the extra time used by the Python interpreter. The student reported that the teacher allowed both AI assistance and C through ctypes.

In the false sharing test, the shared memory starts at an address that is a multiple of 128 bytes. The `padded64` version places counters 8 integers apart. The extra `padded128` version places them 16 integers apart. The actual CPU cache-line size was not confirmed by this test. Workers may run on different types of CPU cores, or at different times.

## Formulas and limits

- **Strong scaling:** speedup S = T1/TP. Efficiency = S/P. A least-squares fit estimates the serial part s using 1/S = s + (1-s)/P.
- **Weak scaling:** the matrix side is ceil(N*sqrt(P)). Work is based on the number of inner cells. `throughput_speedup` = Work(P)/Work(1) * T1/TP. The formula from the PDF is also saved in `pdf_scaled_formula`; it gives a different value. `gustafson_model_speedup` = P-s*(P-1), using s from the strong scaling fit. This is a model prediction, not a separate measurement of the serial part.
- **Karp-Flatt:** the counter test has P*K total work. More processes mean more work. The usual fixed-work Karp-Flatt formula cannot give a clear serial part in this test. The requested calculation is saved in `karp_flatt_fixed_work`. Time ratios and throughput speedups are also saved.
- **Stencil:** each output cell is the average of the centre cell and its four neighbours. Output boundary cells stay zero. Each pass reads the same input. The test does not calculate 80 time steps where each step uses the previous output. The PDF does not give an exact stencil formula or time-step rule.
- **Roofline:** a low number of operations per byte does not prove that main memory limits performance. Some data may come from the cache. The Roofline analysis is incomplete because the hardware limits are not confirmed.

## Missing items and current status

1. The teacher's `verify_outputs.py` was not provided. `verify_local.py` checks the results locally. Its report, `local_verification.json`, is not an official signature. No `verification_receipt.sig` file was created.
2. The operating system blocked some sysctl requests. CPU frequency, cache sizes, and cache-line size are saved as `null` when unavailable. Cache-hit rates and main-memory bandwidth were not measured. Missing hardware values were not invented.
3. Padding did not show a clear improvement in these measurements. The full false sharing problem and recovery required by M2 are not confirmed.
4. The target repository is https://github.com/SultanHello/CSS314-2210-230103379. This lab is prepared in a separate folder, `ParallelComputingLabPracticum_Paradigm`, alongside the earlier tasks. Successful upload was not confirmed at the last check.
5. The code was prepared with AI assistance. The student reported permission from the teacher to use AI and C through ctypes.

## Uploading changes

Use the separate repository copy named `CSS314-2210-230103379`. From its root folder, run:

```sh
git add ParallelComputingLabPracticum_Paradigm
git commit -m "Update parallel computing lab"
git push origin main --tags
```

If there are no new changes to commit, run only the push command.

Commit dates were not changed. The M0-ENV tag points to the commit with the corrected Student ID. The board photo shows a deadline before 13:10. Upload before that time has not been confirmed.

## Upload size

Stencil outputs are stored in two files: `stencil_reference.npz` and `stencil_parallel.npz`. Each is below 25 MB for browser upload. The values are unchanged. Browser upload does not transfer local Git history or tags; use Git to send those.
