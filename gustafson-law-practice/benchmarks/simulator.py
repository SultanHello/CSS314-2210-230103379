"""Task 21: deterministic model predictions, never empirical measurements."""
import argparse,csv,math
from pathlib import Path
from plotting import line_plot

def simulate(counts,alphas):
    for alpha in alphas:
        if not .001<=alpha<=.20:raise ValueError('alpha must be in [0.001,0.20]')
        for p in counts:
            if not 1<=p<=65536:raise ValueError('P must be in [1,65536]')
            a=1/(alpha+(1-alpha)/p);g=p-alpha*(p-1)
            yield dict(p=p,alpha=alpha,amdahl_speedup=a,gustafson_speedup=g,
                       amdahl_efficiency=a/p,gustafson_efficiency=g/p)

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--counts',default='1,2,4,8,16,64,256,1024,4096,16384,65536')
    cli.add_argument('--alphas',default='0.001,0.04,0.08,0.20')
    cli.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'plots')
    args=cli.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    counts=sorted(set(map(int,args.counts.split(','))));alphas=list(map(float,args.alphas.split(',')))
    rows=list(simulate(counts,alphas))
    with (args.output/'simulator.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # One alpha keeps the overlay readable; every alpha/count pair is in the CSV.
    alpha=alphas[0];subset=[r for r in rows if r['alpha']==alpha]
    line_plot(args.output/'model_overlay.pdf','Amdahl and Gustafson predictions',
        f'Model simulation only; same numeric fraction alpha={alpha}; fractions have different reference frames.',[
        dict(label='Ideal linear',points=[(p,p) for p in counts],dash=True),
        dict(label='Gustafson (scaled work)',points=[(r['p'],r['gustafson_speedup']) for r in subset]),
        dict(label='Amdahl (fixed work)',points=[(r['p'],r['amdahl_speedup']) for r in subset])],logx=True)
    print(f'Exported {len(rows)} model rows and PDF/PNG overlay to {args.output}')
if __name__=='__main__':main()
