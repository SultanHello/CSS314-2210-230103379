"""Supplementary C run: amortize process startup without changing defaults."""
from benchmark import benchmark
from exp_c_fixed import run_privatized


def larger_run(p):
    return run_privatized(p, total_increments=200_000_000)


if __name__ == '__main__':
    benchmark(larger_run, 'C supplementary: 200,000,000 increments')
