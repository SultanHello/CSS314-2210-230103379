"""Local numerical checks, NOT the course-provided verifier or an official signature."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; R=ROOT/'results'
sys.path.insert(0,str(ROOT/'src'))
from salt import get_student_params

def digest(a,student_id):
    # Explicit local convention; the course verifier may use another representation.
    return hashlib.sha256(student_id.encode()+np.round(a,6).astype('<f8').tobytes()).hexdigest()

def main():
    p=json.loads((R/'salted_params.json').read_text()); assert p==get_student_params(p['student_id'])
    report={'status':'local_checks_only','official_verifier_run':False,'student_id':p['student_id'],'checks':{}}
    a=np.load(R/'matrix_outputs.npz'); error=float(np.max(np.abs(a['baseline']-a['interchanged'])))
    assert error<=1e-12
    rng=np.random.default_rng(p['seed']); n=p['N']; m=n//2
    left=rng.random((n,n)); right=rng.random((n,n))
    oracle=left[:m,:m] @ right[:m,:m]
    oracle_error=float(np.max(np.abs(a['baseline']-oracle)))
    assert oracle_error<=1e-12, oracle_error
    report['checks']['matrix']={'passed':True,'max_abs_error':error,'numpy_reference_max_abs_error':oracle_error,'baseline_digest':digest(a['baseline'],p['student_id']),
                               'optimized_digest':digest(a['interchanged'],p['student_id'])}
    counters=json.loads((R/'counter_checks.json').read_text())
    assert all(len(x['counts'])==x['P'] and all(v==x['expected_per_worker'] for v in x['counts']) for x in counters)
    report['checks']['counters']={'passed':True,'runs':len(counters)}
    stencil=json.loads((R/'stencil_checks.json').read_text()); assert all(x['passed'] and x['max_abs_error']<=1e-12 for x in stencil)
    a={'reference':np.load(R/'stencil_reference.npz')['reference'],
       'parallel':np.load(R/'stencil_parallel.npz')['parallel']}
    assert np.max(np.abs(a['reference']-a['parallel']))<=1e-12
    report['checks']['stencil']={'passed':True,'runs':len(stencil),'reference_digest':digest(a['reference'],p['student_id']),
                                'parallel_digest':digest(a['parallel'],p['student_id'])}
    (R/'local_verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
