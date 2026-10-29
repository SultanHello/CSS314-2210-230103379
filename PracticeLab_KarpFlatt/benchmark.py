"""Shared repeatable benchmark reporting; standard library only, Python 3.8+."""
import argparse
import json
import statistics


def benchmark(run, name):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--json', dest='json_path')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    samples = {p: [] for p in (1, 2, 4)}
    # Interleave worker counts to reduce drift across repetitions.
    for trial in range(args.repeats):
        for p in (1, 2, 4):
            elapsed = run(p)
            samples[p].append(elapsed)
            print('Trial {} | P={} | Runtime={:.6f}s'.format(trial + 1, p, elapsed), flush=True)
    medians = {p: statistics.median(values) for p, values in samples.items()}
    rows = []
    for p in (1, 2, 4):
        tp = medians[p]
        speedup = medians[1] / tp
        e = None if p == 1 else (p * tp / medians[1] - 1) / (p - 1)
        row = dict(workers=p, seconds=tp, speedup=speedup,
                   efficiency=speedup / p, karp_flatt=e, samples=samples[p])
        rows.append(row)
        print('P={} | Median={:.6f}s | Speedup={:.3f}x | Efficiency={:.2%} | e={}'.format(
            p, tp, speedup, speedup / p, 'N/A' if e is None else '{:.6f}'.format(e)))
    if args.json_path:
        with open(args.json_path, 'w', encoding='utf-8') as stream:
            json.dump(dict(experiment=name, repeats=args.repeats, results=rows), stream, indent=2)
    return rows
