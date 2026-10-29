"""Quick correctness checks, including non-divisible work allocation."""
import math
from exp_a_serial import heavy_math, run_experiment
from exp_b_contention import run_contended
from exp_c_fixed import private_worker, run_privatized

if __name__ == '__main__':
    assert math.isclose(heavy_math(10), sum(math.sin(i)*math.cos(i) for i in range(1,10)))
    assert private_worker(0) == 0
    assert private_worker(17) == 17
    for p in (1, 2, 4):
        assert run_contended(p, 101) > 0
        assert run_privatized(p, 101) > 0
    assert run_experiment(2, total_chunks=2, chunk_size=100) >= .4
    print('PASS: arithmetic, exact totals at P=1/2/4, remainder allocation, serial delay')
