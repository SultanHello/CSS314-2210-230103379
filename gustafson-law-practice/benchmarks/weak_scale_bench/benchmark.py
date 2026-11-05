"""Task 22/23. Real weak-scaling measurements with five independent pool runs.
Use --help. No output data are synthesized. Each process handles k elements.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,datetime,json,multiprocessing as mp,platform,subprocess,sys,time,random
from pathlib import Path
import numpy as np

TASK_BARRIER=None

def initialize(ready,task_barrier):
    global TASK_BARRIER
    TASK_BARRIER=task_barrier
    # Finish imports and a small NumPy warm-up before the parent starts timing.
    kernel((0,1024,1))
    ready.wait(timeout=120)

def kernel(task):
    index,k,iterations=task
    data=np.linspace(.001+index*.01,2+index*.01,k,dtype=np.float64)
    total=0.0
    for j in range(iterations):
        weights=np.sqrt(np.abs(data*5.0)+1.0)
        values=np.exp(-data/4.0)/(1.0+(data/4.0)**2)
        total+=float(np.sum(weights*values))
    return total

def parallel_task(task):
    # Exactly P outstanding tasks block until P distinct workers have started.
    TASK_BARRIER.wait(timeout=120)
    return kernel(task)

def hardware():
    out=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             platform=platform.platform(),machine=platform.machine(),python=sys.version,
             numpy=np.__version__,logical_cores=os.cpu_count(),start_method='spawn',native_threads=1)
    if sys.platform=='darwin':
        raw=subprocess.check_output(['/usr/sbin/system_profiler','SPHardwareDataType','-json'],text=True)
        hw=json.loads(raw)['SPHardwareDataType'][0];top=hw['number_processors'].split()[-1].split(':')
        out.update(cpu=hw['chip_type'],physical_cores=int(top[0]),memory=hw['physical_memory'])
        if len(top)==3:out.update(performance_cores=int(top[1]),efficiency_cores=int(top[2]))
        out['cache_queries']={}
        for key in ('hw.l1dcachesize','hw.l2cachesize','hw.l3cachesize','hw.perflevel0.l2cachesize','hw.perflevel1.l2cachesize'):
            q=subprocess.run(['/usr/sbin/sysctl','-n',key],text=True,capture_output=True)
            out['cache_queries'][key]=dict(value=q.stdout.strip() if q.returncode==0 else None,error=q.stderr.strip() if q.returncode else None)
    else:
        import psutil
        out.update(cpu=platform.processor(),physical_cores=psutil.cpu_count(logical=False))
    out['unmeasured']=['frequency','temperature','power','DRAM traffic','memory channel count','NUMA placement']
    return out

def run(k,iterations,counts,repeats,folder):
    ctx=mp.get_context('spawn');info=hardware()
    info.update(elements_per_worker=k,iterations=iterations,repetitions=repeats,process_counts=counts,
                measured_scope='ready pool dispatch/barrier through result collection; allocation and kernel included; spawn/import/teardown excluded',
                serial_scope='same P task inputs executed sequentially in parent; kernel warmup before trials',
                independent_runs='fresh ready pool for every configuration/repeat',schedule_seed=230103379)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'telemetry.json').write_text(json.dumps(info,indent=2))
    schedule=[(p,r) for r in range(1,repeats+1) for p in counts];random.Random(230103379).shuffle(schedule)
    kernel((0,k,iterations))
    fields=['p','repeat','total_elements','elements_per_worker','iterations','t_parallel_s','t_serial_s','speedup','checksum','serial_checksum','startup_ready_s','serial_first']
    with (folder/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p,r in schedule:
            tasks=[(i,k,iterations) for i in range(p)]
            ready=ctx.Barrier(p+1);task_barrier=ctx.Barrier(p)
            begin=time.perf_counter()
            with ctx.Pool(p,initializer=initialize,initargs=(ready,task_barrier)) as pool:
                ready.wait(timeout=120);startup=time.perf_counter()-begin
                def serial():
                    start=time.perf_counter();value=sum(map(kernel,tasks));return time.perf_counter()-start,value
                def parallel():
                    start=time.perf_counter();values=pool.map(parallel_task,tasks,chunksize=1);return time.perf_counter()-start,sum(values)
                first=bool(r%2)
                if first:ts,cs=serial();tp,cp=parallel()
                else:tp,cp=parallel();ts,cs=serial()
            if not np.isclose(cs,cp,rtol=1e-12,atol=1e-8):raise RuntimeError('Numerical mismatch')
            row=dict(p=p,repeat=r,total_elements=p*k,elements_per_worker=k,iterations=iterations,
                     t_parallel_s=tp,t_serial_s=ts,speedup=ts/tp,checksum=cp,serial_checksum=cs,
                     startup_ready_s=startup,serial_first=first)
            writer.writerow(row);f.flush()
            print(f'P={p:2d} run={r}: parallel={tp:.4f}s serial={ts:.4f}s speedup={ts/tp:.3f}',flush=True)
    return info

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--k',type=int,default=960000)
    parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--counts',default=None);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();maxp=os.cpu_count() or 1
    # Full hardware-concurrency range by default, not just powers of two.
    counts=list(range(1,maxp+1)) if args.counts is None else sorted(set(map(int,args.counts.split(','))))
    if args.k<1 or args.iterations<1 or args.repeats<1 or any(p<1 or p>maxp for p in counts):parser.error('Invalid workload/counts')
    run(args.k,args.iterations,counts,args.repeats,args.output)
if __name__=='__main__':mp.freeze_support();main()
