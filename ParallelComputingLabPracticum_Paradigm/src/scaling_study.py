import ctypes as ct
import multiprocessing as mp
import math, statistics, time
import numpy as np
from common import lib, ptr, read, save, tsv, RESULTS
PASSES=80

def worker(a,b,n,lo,hi,barrier,starts,ends,w):
    l=lib(); aa=np.frombuffer(a,dtype=np.float64); bb=np.frombuffer(b,dtype=np.float64)
    l.stencil(ptr(aa),ptr(bb),n,lo,hi,1)
    barrier.wait(timeout=120); starts[w]=time.perf_counter_ns()
    l.stencil(ptr(aa),ptr(bb),n,lo,hi,PASSES)
    ends[w]=time.perf_counter_ns()

def experiment(ctx,n,p,seed):
    a=ctx.RawArray(ct.c_double,n*n); b=ctx.RawArray(ct.c_double,n*n)
    aa=np.frombuffer(a,dtype=np.float64).reshape(n,n)
    bb=np.frombuffer(b,dtype=np.float64).reshape(n,n)
    aa[:]=np.random.default_rng(seed).random((n,n))
    barrier=ctx.Barrier(p+1); starts=ctx.RawArray(ct.c_uint64,p); ends=ctx.RawArray(ct.c_uint64,p)
    bounds=np.linspace(1,n-1,p+1,dtype=int)
    jobs=[ctx.Process(target=worker,args=(a,b,n,int(bounds[w]),int(bounds[w+1]),barrier,starts,ends,w)) for w in range(p)]
    for job in jobs: job.start()
    barrier.wait(timeout=120)
    for job in jobs:
        job.join(timeout=120)
        if job.exitcode!=0: raise RuntimeError(f'Stencil worker failed: {job.exitcode}')
    ref=(aa[1:-1,1:-1]+aa[1:-1,:-2]+aa[1:-1,2:]+aa[:-2,1:-1]+aa[2:,1:-1])*0.2
    err=float(np.max(np.abs(bb[1:-1,1:-1]-ref)))
    if err>1e-12: raise AssertionError(err)
    checksum=float(bb.sum())
    return max(ends)-min(starts),err,checksum,bb.copy()

def main():
    lib(); ctx=mp.get_context('spawn'); params=read('salted_params.json'); host=read('host_profile.json')
    maximum=host['physical_cores'] or 1; ps=[1]; p=2
    while p<=maximum: ps.append(p); p*=2
    if maximum not in ps: ps.append(maximum)
    rows=[]; verification=[]; baseline=None; largest=None
    for study in ['strong','weak']:
        base=None
        for p in ps:
            n=params['N'] if study=='strong' else math.ceil(params['N']*math.sqrt(p))
            times=[]
            for run in range(3):
                elapsed,err,checksum,out=experiment(ctx,n,p,params['seed']); times.append(elapsed)
                verification.append(dict(study=study,P=p,N=n,run=run+1,max_abs_error=err,checksum=checksum,passed=True))
            if p==1: base=statistics.median(times)
            t=statistics.median(times); speed=base/t; ratio=((n-2)/(params['N']-2))**2
            row=dict(study=study,P=p,N=n,passes=PASSES,run1_ns=times[0],run2_ns=times[1],run3_ns=times[2],
                     median_time_ns=t,speedup=speed if study=='strong' else '',
                     efficiency_pct=100*speed/p if study=='strong' else '',
                     work_ratio=ratio,weak_efficiency_pct=100*speed if study=='weak' else '',
                     throughput_speedup=ratio*speed,
                     pdf_scaled_formula=p*speed/ratio if study=='weak' else '',
                     amdahl_s=(1/speed-1/p)/(1-1/p) if study=='strong' and p>1 else '')
            rows.append(row)
            if study=='strong' and p==1: baseline=out
            if study=='strong' and p==maximum: largest=out
            print('Stencil',study,p,n,flush=True)
    np.savez_compressed(RESULTS/'stencil_reference.npz',reference=baseline)
    np.savez_compressed(RESULTS/'stencil_parallel.npz',parallel=largest)
    save('stencil_checks.json',verification)
    strong=[r for r in rows if r['study']=='strong' and r['P']>1]
    x=np.array([1-1/r['P'] for r in strong]); y=np.array([1/r['speedup']-1/r['P'] for r in strong])
    s=float(np.dot(x,y)/np.dot(x,x)) if len(x) else None
    constrained=min(1,max(0,s)) if s is not None else None
    for row in rows:
        row['gustafson_model_speedup']=row['P']-constrained*(row['P']-1) if constrained is not None and row['study']=='weak' else ''
    tsv('scaling_metrics.tsv',rows)
    summary={'stencil':'out[i,j]=0.2*(center+left+right+up+down); fixed input on every pass',
             'flops_per_update':5,'bytes_per_update_no_reuse':48,'arithmetic_intensity_no_reuse':5/48,
             'bytes_per_update_ideal_reuse_with_write_allocate':24,'arithmetic_intensity_ideal_reuse':5/24,
             'peak_memory_bandwidth_bytes_s':None,'peak_compute_flops_s':None,
             'classification':'Low arithmetic intensity suggests memory traffic sensitivity; no measured/verified DRAM bandwidth or compute ceiling, so a definitive Roofline bound is unavailable.',
             'cache_telemetry':{k:v for k,v in host.items() if k in ['cache_line_bytes','l1_data_bytes','l2_bytes','l3_bytes']},
             'amdahl_s_unconstrained_fit':s,'amdahl_s_constrained_fit':constrained,
             'amdahl_asymptotic_speedup':1/constrained if constrained and constrained>0 else None,
             'notes':['Worker startup and one warmup pass excluded; timers span earliest worker start to latest completion. Scheduling skew remains included.',
                      'Apple M4 has performance and efficiency cores; equal-core scaling models are approximations.',
                      'Weak study logs PDF expression separately; normalized throughput speedup is work_ratio*T1/TP.',
                      'No hardware performance counters were available; cache-hit rates are not fabricated.']}
    save('roofline_summary.json',summary)
if __name__=='__main__': main()
