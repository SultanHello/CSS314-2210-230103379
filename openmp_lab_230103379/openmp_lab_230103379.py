import os

os.environ["NUMBA_THREADING_LAYER"] = "omp"

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import numba as nb
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads, types
from numba.extending import intrinsic

STUDENT_NAME = "Assimbek Sultan"
STUDENT_ID = 230103379
THREAD_COUNTS = (1, 2, 4, 8)

N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

DIAGNOSIS_T1 = (
    "A histogram update has three steps: read the count, add one, and write the "
    "result. If two threads read the same count before either writes, one update can "
    "be lost. In ten runs with eight threads, the unsafe version lost about three "
    "million counts per run. Atomic integer updates would protect the counts, but "
    "floating-point sums could still differ because the order of addition affects "
    "rounding. The safe version gives each thread its own bins and stores the "
    "weights as integer units. It merges the integer totals before converting back "
    "to floating point, and the results match bit for bit at 1, 2, 4, and 8 threads."
)
DIAGNOSIS_T2 = (
    "In the unpadded array, different threads can write to different values on the "
    "same cache line. Under MESI-style cache coherence, a core needs write ownership "
    "of that line, so a write can invalidate copies held by other cores. Padding "
    "places the active slots 320 bytes apart, and both versions still write to "
    "memory inside the loop. Both versions produced the same values at every thread "
    "count. With eight threads, the unpadded version took 56.44 ms and the padded "
    "version took 55.24 ms, a speedup of about 1.02x in this run."
)
DIAGNOSIS_T3 = (
    "Static scheduling gives each thread a fixed range of rows, but some rows "
    "require more operations than others. The dynamic scheduler mixes light and "
    "heavy rows, and threads use an atomic counter to claim batches of 16 rows. Each "
    "thread stops claiming work when it reaches its operation target or the queue is "
    "empty. With eight threads, both methods returned exactly the same scores. The "
    "dynamic imbalance ratio was 1.014, below the required 1.15, while the static "
    "ratio was slightly lower at 1.005. The dynamic kernel itself took 6.99 ms "
    "versus 7.76 ms for the static kernel."
)


@njit(parallel=True)
def _omp_probe():
    hits = np.zeros(get_num_threads(), dtype=np.int64)
    for _ in prange(64):
        hits[get_thread_id()] += 1
    return hits


@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.size):
        b = indices[i]
        hist[b] += 1
        w_sums[b] += weights[i]
    return hist, w_sums


@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    threads = get_num_threads()
    counts = np.zeros((threads, num_bins), dtype=np.int64)
    upper = np.zeros((threads, num_bins), dtype=np.int64)
    lower = np.zeros((threads, num_bins), dtype=np.int64)
    hits = np.zeros(threads, dtype=np.int64)
    # Binary64 weights in [0.25, 2.50] are exact multiples of 2**-54.
    # Two limbs avoid int64 overflow at the seeded input size.
    for i in prange(indices.size):
        tid = get_thread_id()
        b = indices[i]
        units = np.int64(weights[i] * 18014398509481984.0)
        counts[tid, b] += 1
        upper[tid, b] += units >> 32
        lower[tid, b] += units & 0xFFFFFFFF
        hits[tid] += 1
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        hi = np.int64(0)
        lo = np.int64(0)
        for tid in range(threads):
            hist[b] += counts[tid, b]
            hi += upper[tid, b]
            lo += lower[tid, b]
        hi += lo >> 32
        lo &= 0xFFFFFFFF
        w_sums[b] = hi * (2.0 ** -22) + lo * (2.0 ** -54)
    return hist, w_sums, hits


@intrinsic
def _store_i64(typingctx, array, index, value):
    signature = types.void(array, index, value)

    def codegen(context, builder, sig, args):
        view = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = builder.gep(view.data, [args[1]])
        builder.store_atomic(args[2], ptr, "monotonic", align=8)
        return context.get_dummy_value()

    return signature, codegen


@intrinsic
def _claim_i64(typingctx, array):
    signature = types.int64(array)

    def codegen(context, builder, sig, args):
        view = context.make_array(sig.args[0])(context, builder, args[0])
        one = context.get_constant(types.int64, 1)
        return builder.atomic_rmw("add", view.data, one, "monotonic")

    return signature, codegen


@njit(inline="always")
def _state_worker(flat, slot, iters, seed_val, tid):
    x = np.int64(seed_val + tid * 1315423911)
    acc = np.int64(0)
    for _ in range(iters):
        x = (x * np.int64(6364136223846793005) +
             np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
        acc ^= x >> 17
        _store_i64(flat, slot, acc)


@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    threads = get_num_threads()
    state = np.empty(threads, dtype=np.int64)
    hits = np.zeros(threads, dtype=np.int64)
    for _ in prange(threads):
        tid = get_thread_id()
        _state_worker(state, tid, iters, seed_val, tid)
        hits[tid] += iters
    return state, hits


@njit(parallel=True)
def padded_state_omp(iters, seed_val):
    threads = get_num_threads()
    state = np.zeros((threads, PAD_STRIDE), dtype=np.int64)
    flat = state.reshape(threads * PAD_STRIDE)
    hits = np.zeros(threads, dtype=np.int64)
    for _ in prange(threads):
        tid = get_thread_id()
        _state_worker(flat, tid * PAD_STRIDE, iters, seed_val, tid)
        hits[tid] += iters
    return state, hits


@njit(inline="always")
def _row_score(row, steps):
    total = 0.0
    for step in range(steps):
        x = (row + 1) * 0.0001 + step * 0.00001
        total += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return total


@njit(parallel=True)
def static_skew_omp(work_units):
    out = np.empty(work_units.size, dtype=np.float64)
    counts = np.zeros((get_num_threads(), PAD_STRIDE), dtype=np.int64)
    for r in prange(work_units.size):
        steps = work_units[r]
        out[r] = _row_score(r, steps)
        counts[get_thread_id(), 0] += steps
    return out, counts[:, 0].copy()


@njit(parallel=True)
def balanced_skew_omp(work_units, order, chunk, goal):
    out = np.empty(work_units.size, dtype=np.float64)
    threads = get_num_threads()
    counts = np.zeros((threads, PAD_STRIDE), dtype=np.int64)
    cursor = np.zeros(1, dtype=np.int64)
    for _ in prange(threads):
        tid = get_thread_id()
        done = np.int64(0)
        while done < goal:
            begin = _claim_i64(cursor) * chunk
            if begin >= order.size:
                break
            end = min(begin + chunk, order.size)
            for i in range(begin, end):
                row = order[i]
                steps = work_units[row]
                out[row] = _row_score(row, steps)
                done += steps
        counts[tid, 0] = done
    return out, counts[:, 0].copy()


@njit(parallel=True)
def sor_red_black_step(grid, omega, color, delta, hits):
    h, w = grid.shape
    for row in prange(1, h - 1):
        r = np.int64(row)
        tid = get_thread_id()
        start = 1 + ((r + 1 + color) % 2)
        local_max = delta[tid, 0]
        updated = 0
        for c in range(start, w - 1, 2):
            old = grid[r, c]
            neighbors = grid[r - 1, c] + grid[r + 1, c] + grid[r, c - 1] + grid[r, c + 1]
            new = (1.0 - omega) * old + 0.25 * omega * neighbors
            grid[r, c] = new
            local_max = max(local_max, abs(new - old))
            updated += 1
        delta[tid, 0] = local_max
        hits[tid, 0] += updated


@njit
def solve_pde_omp(initial, omega, iterations):
    grid = initial.copy()
    threads = get_num_threads()
    delta = np.zeros((threads, PAD_STRIDE), dtype=np.float64)
    hits = np.zeros((threads, PAD_STRIDE), dtype=np.int64)
    history = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        for tid in range(threads):
            delta[tid, 0] = 0.0
        sor_red_black_step(grid, omega, 0, delta, hits)
        sor_red_black_step(grid, omega, 1, delta, hits)
        maximum = 0.0
        for tid in range(threads):
            maximum = max(maximum, delta[tid, 0])
        history[i] = maximum
    return grid, history, hits[:, 0].copy()


def timed(fn, *args, repeat=3):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn(*args)
        samples.append((time.perf_counter() - start) * 1000)
    return value, statistics.median(samples)


def participation(hits, threads, total):
    assert hits.shape == (threads,) and np.all(hits > 0), hits
    assert int(hits.sum()) == total, hits
    return hits.tolist()


def run_histogram(rng, smoke):
    n = 32_768 if smoke else N_T1
    indices = rng.integers(0, BINS_T1, size=n, dtype=np.int32)
    weights = rng.uniform(0.25, 2.50, size=n).astype(np.float64)
    set_num_threads(1)
    ref_hist, ref_weights = buggy_hist_omp(indices, weights, BINS_T1)
    exact_hist, exact_weights, _ = safe_hist_omp(indices, weights, BINS_T1)
    assert np.array_equal(ref_hist, exact_hist)
    np.testing.assert_allclose(exact_weights, ref_weights, rtol=1e-12, atol=1e-9)
    set_num_threads(8)
    races = []
    for i in range(10):
        hist, sums = buggy_hist_omp(indices, weights, BINS_T1)
        row = {"run": i + 1, "lost_counts": n - int(hist.sum()),
               "max_count_error": int(np.max(np.abs(hist - ref_hist))),
               "max_weight_error": float(np.max(np.abs(sums - ref_weights)))}
        races.append(row)
        print(f"T1 unsafe: {row}", flush=True)
    safe = []
    for threads in THREAD_COUNTS:
        set_num_threads(threads)
        (hist, sums, hits), ms = timed(safe_hist_omp, indices, weights, BINS_T1)
        assert np.array_equal(hist, exact_hist)
        assert np.array_equal(sums.view(np.uint64), exact_weights.view(np.uint64))
        safe.append({"threads": threads, "median_ms": ms,
                     "worker_hits": participation(hits, threads, n)})
        print(f"T1 safe {threads} threads: {ms:.3f} ms, hits={hits.tolist()}", flush=True)
    return hist, sums, {"elements": n, "bins": BINS_T1, "races": races, "safe": safe}


def run_false_sharing(smoke):
    iters = 30_000 if smoke else ITERS_T2
    stats = []
    for threads in THREAD_COUNTS:
        set_num_threads(threads)
        unpadded_state_omp(64, STUDENT_ID)
        padded_state_omp(64, STUDENT_ID)
        (a, a_hits), a_ms = timed(unpadded_state_omp, iters, STUDENT_ID)
        (b, b_hits), b_ms = timed(padded_state_omp, iters, STUDENT_ID)
        assert np.array_equal(a, b[:, 0])
        stats.append({"threads": threads, "unpadded_ms": a_ms, "padded_ms": b_ms,
                      "ratio": a_ms / b_ms,
                      "unpadded_worker_hits": participation(a_hits, threads, threads * iters),
                      "padded_worker_hits": participation(b_hits, threads, threads * iters)})
        print(f"T2 {threads} threads: {a_ms:.3f} / {b_ms:.3f} ms, ratio={a_ms/b_ms:.3f}", flush=True)
    return b[:, 0].copy(), {"iterations_per_worker": iters,
                          "pad_stride": PAD_STRIDE, "runs": stats,
                          "speedup_8_threads": stats[-1]["ratio"]}


def chunk_plan(units, threads):
    rows = units.size
    # Interleave the input from the ends of an ordering by estimated row cost.
    sorted_rows = np.argsort(units, kind="stable")
    order = np.empty(rows, dtype=np.int64)
    order[::2] = sorted_rows[:(rows + 1) // 2]
    order[1::2] = sorted_rows[-(rows // 2):][::-1]
    total = int(units.sum())
    goal = (total + threads - 1) // threads
    chunk = min(16, rows)
    while True:
        maximum_batch = int(np.max(np.add.reduceat(units[order], np.arange(0, rows, chunk))))
        upper = goal + maximum_batch - 1
        lower = total - (threads - 1) * upper
        if lower > 0 and upper / lower < 1.15:
            return order, chunk, goal
        if chunk == 1:
            raise ValueError("Insufficient workload to bound imbalance with unit-size chunks.")
        chunk //= 2


def run_scheduler(smoke):
    rows = 4096 if smoke else ROWS_T3
    indices = np.arange(rows, dtype=np.int64)
    units = 40 + ((indices * 7919 + STUDENT_ID) % SKEW_MOD) ** 2
    total, stats = int(units.sum()), []
    for threads in THREAD_COUNTS:
        set_num_threads(threads)
        order, chunk, goal = chunk_plan(units, threads)
        static_skew_omp(units[:64])
        balanced_skew_omp(units[:64], np.arange(64, dtype=np.int64), 2, 2**30)
        (a, a_hits), a_ms = timed(static_skew_omp, units)
        (b, b_hits), b_ms = timed(balanced_skew_omp, units, order, chunk, goal)
        error = float(np.max(np.abs(a - b)))
        assert error < 1e-11, error
        ratio_a = float(a_hits.max() / a_hits.min())
        ratio_b = float(b_hits.max() / b_hits.min())
        assert ratio_b < 1.15, ratio_b
        stats.append({"threads": threads, "static_ms": a_ms, "balanced_ms": b_ms,
                      "static_imbalance": ratio_a, "balanced_imbalance": ratio_b,
                      "chunk_rows": chunk, "max_abs_diff": error,
                      "static_worker_ops": participation(a_hits, threads, total),
                      "balanced_worker_ops": participation(b_hits, threads, total)})
        print(f"T3 {threads} threads: {a_ms:.3f} / {b_ms:.3f} ms, imbalance {ratio_a:.3f} / {ratio_b:.3f}", flush=True)
    return b, {"rows": rows, "runs": stats}


def initial_grid(h, w):
    grid = np.zeros((h, w), dtype=np.float64)
    grid[0, :] = 100.0
    grid[-1, :] = -50.0
    # Apply the side edges last, as in the order specified; they own the corners.
    grid[:, 0] = 25.0 * np.sin(np.linspace(0, np.pi, h))
    grid[:, -1] = 75.0 * np.cos(np.linspace(0, np.pi / 2, h))
    return grid


def run_sor(smoke):
    h, w, iterations = (33, 41, 20) if smoke else (GRID_H, GRID_W, 600)
    initial = initial_grid(h, w)
    set_num_threads(1)
    solve_pde_omp(initial[:10, :10].copy(), OMEGA_SOR, 1)
    stats, reference = [], None
    for threads in THREAD_COUNTS:
        set_num_threads(threads)
        (grid, residual, hits), ms = timed(solve_pde_omp, initial, OMEGA_SOR, iterations)
        if reference is None:
            reference, ref_ms = grid, ms
        diff = float(np.max(np.abs(grid - reference)))
        assert diff < 1e-9, diff
        assert np.array_equal(grid[[0, -1], :], initial[[0, -1], :])
        assert np.array_equal(grid[:, [0, -1]], initial[:, [0, -1]])
        assert np.isfinite(residual).all() and 0 < residual[-1] < residual[0]
        stats.append({"threads": threads, "median_ms": ms, "speedup": ref_ms / ms,
                      "max_abs_diff": diff, "final_max_delta": float(residual[-1]),
                      "worker_updates": participation(hits, threads, iterations * (h - 2) * (w - 2))})
        print(f"T4 {threads} threads: {ms:.3f} ms, speedup={ref_ms/ms:.3f}, max_delta={residual[-1]:.6g}", flush=True)
    return grid, {"shape": [h, w], "iterations": iterations,
                  "omega": float(OMEGA_SOR), "max_delta_history_8_threads": residual.tolist(),
                  "runs": stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Reduced test; no submission JSON")
    args = parser.parse_args()
    assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, "Set a valid numeric STUDENT_ID!"
    cores = os.cpu_count() or 1
    assert cores >= 4, "Local multi-core CPU (>=4 logical cores) required; no Colab!"
    if nb.config.NUMBA_NUM_THREADS < 8:
        raise SystemExit("The required eight-thread run is unavailable to this Python process.")
    try:
        set_num_threads(8)
        probe = _omp_probe()
    except ValueError as exc:
        if "threading layer" not in str(exc).lower():
            raise
        raise SystemExit("Numba could not start OpenMP. Check 'numba -s'; on Apple Silicon "
                         "use a native environment with numba, numpy and llvm-openmp.") from exc
    assert nb.threading_layer() == "omp", f"Expected 'omp', got {nb.threading_layer()}"
    participation(probe, 8, 64)
    print(f"Backend: {nb.threading_layer()} | Logical Cores: {cores} | Student: {STUDENT_ID}", flush=True)
    rng = np.random.default_rng(STUDENT_ID)
    hist_t1, wsums_t1, t1 = run_histogram(rng, args.smoke)
    state_t2, t2 = run_false_sharing(args.smoke)
    out_t3, t3 = run_scheduler(args.smoke)
    grid_t4, t4 = run_sor(args.smoke)
    if args.smoke:
        print("Smoke test passed. No submission JSON was created.")
        return

    set_num_threads(8 if cores >= 8 else cores)
    for i, diagnosis in enumerate((DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3), start=1):
        assert isinstance(diagnosis, str) and len(diagnosis.strip()) >= 80, f"Complete DIAGNOSIS_T{i}!"
    checksum_blob = (
        f"{STUDENT_ID}|"
        f"{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
        f"{int(np.bitwise_xor.reduce(state_t2))}|"
        f"{float(np.sum(out_t3)):.6f}|"
        f"{float(np.sum(grid_t4)):.6f}"
    )
    lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()
    bundle = {
        "instructor": "Instructor - Sufyan bin Uzayr",
        "student_name": STUDENT_NAME,
        "student_id": STUDENT_ID,
        "omp_layer": nb.threading_layer(),
        "cpu_cores": cores,
        "verification_digest": lab_digest,
        "diagnoses": {"T1": DIAGNOSIS_T1, "T2": DIAGNOSIS_T2, "T3": DIAGNOSIS_T3},
        "telemetry": {"T1": t1, "T2": t2, "T3": t3, "T4": t4},
    }
    payload = json.dumps(bundle, indent=2, allow_nan=False)
    destination = Path(__file__).with_name(f"openmp_lab_{STUDENT_ID}.json")
    dest_temp = destination.with_suffix(".tmp")
    dest_temp.write_text(payload + "\n", encoding="utf-8")
    dest_temp.replace(destination)
    print("=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
    print(payload)
    print(f"Saved {destination.resolve()}")


if __name__ == "__main__":
    main()
