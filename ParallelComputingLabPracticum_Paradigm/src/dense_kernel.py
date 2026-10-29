import time, statistics
import numpy as np
from common import lib, ptr, read, save, RESULTS

def main():
    p=read('salted_params.json'); n=p['N']; m=n//2; rng=np.random.default_rng(p['seed'])
    a=rng.random((n,n)); b=rng.random((n,n)); l=lib(); outputs=[]; results={}
    for order,name in enumerate(['ijk','ikj']):
        c=np.zeros((n,n)); l.matmul(ptr(a),ptr(b),ptr(c),n,m,order); times=[]
        for _ in range(3):
            t=time.perf_counter_ns(); l.matmul(ptr(a),ptr(b),ptr(c),n,m,order); times.append(time.perf_counter_ns()-t)
        outputs.append(c[:m,:m].copy()); results[name]={'runs_ns':times,'median_ns':statistics.median(times)}
    diff=float(np.max(np.abs(outputs[0]-outputs[1])))
    np.savez(RESULTS/'matrix_outputs.npz',baseline=outputs[0],interchanged=outputs[1])
    results.update(N=n,M=m,speedup=results['ijk']['median_ns']/results['ikj']['median_ns'],
                   max_abs_error=diff,passed=bool(diff<=1e-12),backend='ctypes C loops, identical scalar addition order; no BLAS')
    save('loop_interchange.json',results); assert results['passed']; print('Matrix benchmark complete',flush=True)
if __name__=='__main__': main()
