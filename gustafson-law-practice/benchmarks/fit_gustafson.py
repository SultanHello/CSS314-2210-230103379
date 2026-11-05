"""Tasks 23/24: OLS constrained line through S(1)=1; report unconstrained alpha.
S(P)=P-alpha*(P-1). Startup is excluded in the input measurements.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from plotting import line_plot

def summarize(rows):
    summary=[]
    for p in sorted(set(int(r['p']) for r in rows)):
        group=[r for r in rows if int(r['p'])==p];n=len(group)
        item=dict(p=p,n=n)
        for field in ('t_parallel_s','t_serial_s','speedup'):
            values=np.array([float(r[field]) for r in group]);mean=float(values.mean())
            sd=float(values.std(ddof=1)) if n>1 else 0.0
            item[field+'_mean']=mean;item[field+'_sd']=sd;item[field+'_sem']=sd/math.sqrt(n)
        summary.append(item)
    return summary

def fit(summary):
    x=np.array([r['p']-1 for r in summary],dtype=float)
    p=x+1;s=np.array([r['speedup_mean'] for r in summary]);y=p-s
    if np.dot(x,x)==0:raise ValueError('Need a process count above one')
    alpha=float(np.dot(x,y)/np.dot(x,x))
    residual=s-(p-alpha*x);sse=float(np.dot(residual,residual));dof=len(summary)-1
    stderr=math.sqrt(sse/dof/float(np.dot(x,x))) if dof>0 else None
    sst=float(np.sum((s-s.mean())**2))
    return dict(alpha_unconstrained=alpha,alpha_physical_projection=max(0,min(1,alpha)),
                alpha_ols_standard_error=stderr,sse=sse,r_squared=1-sse/sst if sst else None,
                method='equal-weight OLS of group mean speedup; theoretical intercept fixed S(1)=1',
                warning='Apparent overhead coefficient, not an identified algorithmic serial fraction. Same-alpha Amdahl curve is illustrative, not a measured strong-scaling result.')

def main():
    root=Path(__file__).resolve().parents[1]
    cli=argparse.ArgumentParser();cli.add_argument('--input',type=Path,default=root/'benchmarks/results.csv')
    cli.add_argument('--output',type=Path,default=root/'plots');args=cli.parse_args()
    rows=list(csv.DictReader(args.input.open()));summary=summarize(rows);model=fit(summary)
    with (args.input.parent/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (args.input.parent/'fit.json').write_text(json.dumps(model,indent=2))
    alpha=model['alpha_unconstrained'];maximum=max(r['p'] for r in summary)
    dense=np.linspace(1,maximum,150)
    amdahl_alpha=model['alpha_physical_projection']
    line_plot(args.output/'scaling_comparison.pdf','Empirical weak scaling and model curves',
        f'Apple M4; five runs/count; error bars = speedup SEM; apparent OLS alpha={alpha:.4f}',[
        dict(label='Ideal linear S=P',points=[(p,p) for p in dense],dash=True),
        dict(label='Gustafson OLS fit',points=[(p,p-alpha*(p-1)) for p in dense]),
        dict(label='Measured mean +/- SEM',points=[(r['p'],r['speedup_mean'],r['speedup_sem']) for r in summary],markers=True,connect=False),
        dict(label='Amdahl (same alpha; illustrative)',points=[(p,1/(amdahl_alpha+(1-amdahl_alpha)/p)) for p in dense],dash=True)])
    print(json.dumps(model,indent=2))
if __name__=='__main__':main()
