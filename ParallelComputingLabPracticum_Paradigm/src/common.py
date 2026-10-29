from pathlib import Path
import ctypes as ct
import csv, json, subprocess, sys, platform
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

def save(name, value):
    (RESULTS/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def read(name):
    return json.loads((RESULTS/name).read_text())

def tsv(name, rows):
    with (RESULTS/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(rows)

def lib():
    source=ROOT/'src/kernels.c'
    target=ROOT/'src'/('kernels.dylib' if sys.platform=='darwin' else 'kernels.so')
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['cc','-O2','-fno-fast-math','-ffp-contract=off','-shared',
                        '-fPIC',str(source),'-o',str(target)],check=True)
    l=ct.CDLL(str(target))
    ptr=ct.POINTER(ct.c_double)
    l.stride_sum.argtypes=[ptr,ct.c_size_t,ct.c_size_t]; l.stride_sum.restype=ct.c_double
    l.matmul.argtypes=[ptr,ptr,ptr,ct.c_int,ct.c_int,ct.c_int]
    l.stencil.argtypes=[ptr,ptr,ct.c_int,ct.c_int,ct.c_int,ct.c_int]
    for name in ['count_shared','count_local']:
        getattr(l,name).argtypes=[ct.POINTER(ct.c_int64),ct.c_size_t,ct.c_int64]
    return l

def ptr(a): return a.ctypes.data_as(ct.POINTER(ct.c_double))
