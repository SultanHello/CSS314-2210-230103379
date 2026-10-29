import ctypes as ct
import multiprocessing as mp
import statistics, time
from common import lib, tsv, save
K=5_000_000

def worker(a,index,iterations,mode,barrier,starts,ends,w):
    l=lib(); f=l.count_local if mode=='privatized' else l.count_shared
    barrier.wait(timeout=120); starts[w]=time.perf_counter_ns()
    f(a,index,iterations)
    ends[w]=time.perf_counter_ns()

def experiment(ctx,p,mode):
    # RawArray has no global lock. Align base to 128 bytes for Apple cache lines;
    # requested 64-byte padding is measured separately, not silently changed.
    raw=ctx.RawArray(ct.c_int64, p*16+32)
    offset=((-ct.addressof(raw))%128)//8
    spacing={'unpadded':1,'padded64':8,'padded128':16,'privatized':1}[mode]
    starts=ctx.RawArray(ct.c_uint64,p); ends=ctx.RawArray(ct.c_uint64,p); barrier=ctx.Barrier(p+1)
    jobs=[ctx.Process(target=worker,args=(raw,offset+w*spacing,K,mode,barrier,starts,ends,w)) for w in range(p)]
    for job in jobs: job.start()
    barrier.wait(timeout=120)
    for job in jobs:
        job.join(timeout=120)
        if job.exitcode!=0: raise RuntimeError(f'Counter worker failed: {job.exitcode}')
    values=[raw[offset+w*spacing] for w in range(p)]
    assert values==[K]*p
    return max(ends)-min(starts),values

def main():
    lib(); ctx=mp.get_context('spawn'); rows=[]; checks=[]
    for mode in ['unpadded','padded64','padded128','privatized']:
        timings={}
        for p in [1,2,4,8]:
            times=[]
            for run in range(3):
                elapsed,values=experiment(ctx,p,mode); times.append(elapsed)
                checks.append({'variant':mode,'P':p,'run':run+1,'counts':values,'expected_per_worker':K})
            timings[p]=statistics.median(times)
            rows.append(dict(variant=mode,P=p,iterations_per_worker=K,total_iterations=p*K,
                             run1_ns=times[0],run2_ns=times[1],run3_ns=times[2],median_time_ns=timings[p],
                             raw_time_ratio=0,throughput_speedup=0,karp_flatt_fixed_work='',weak_efficiency=0))
            print('Counters',mode,p,flush=True)
        for row in rows[-4:]:
            p=row['P']; t=timings[p]; t1=timings[1]
            row['raw_time_ratio']=t1/t
            row['throughput_speedup']=p*t1/t
            row['weak_efficiency']=t1/t
            # K per worker means growing work. Fixed-work Karp-Flatt isn't valid here.
            # Preserve requested plug-in calculation explicitly, without calling it a serial fraction.
            row['karp_flatt_fixed_work']=(p*t/t1-1)/(p-1) if p>1 else ''
    tsv('false_sharing_karp_flatt.tsv',rows)
    save('counter_checks.json',checks)
if __name__=='__main__': main()
