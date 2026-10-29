import time, statistics
import numpy as np
from common import lib, ptr, read, tsv

def main():
    p=read('salted_params.json'); a=np.ones(p['buffer_elements'],dtype=np.float64); l=lib(); rows=[]
    for s in [1,2,4,8,16,32,64,128]:
        count=(len(a)+s-1)//s
        for _ in range(5): l.stride_sum(ptr(a),len(a),s)
        times=[]
        for run in range(5):
            t=time.perf_counter_ns(); total=l.stride_sum(ptr(a),len(a),s); elapsed=time.perf_counter_ns()-t
            assert total==count; times.append(elapsed)
            rows.append(dict(stride=s,run=run+1,time_ns=elapsed,elements=count,ns_per_element=elapsed/count,median_time_ns=0))
        for row in rows[-5:]: row['median_time_ns']=statistics.median(times)
    tsv('stride_results.tsv',rows); print('Stride benchmark complete',flush=True)
if __name__=='__main__': main()
