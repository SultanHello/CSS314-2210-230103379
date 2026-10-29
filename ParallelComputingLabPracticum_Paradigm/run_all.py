import subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
if len(sys.argv)!=2: raise SystemExit('Usage: python3 run_all.py STUDENT_ID')
for script,args in [('profile_env.py',['--student-id',sys.argv[1]]),('stride_bench.py',[]),('dense_kernel.py',[]),('false_sharing.py',[]),('scaling_study.py',[])]:
    subprocess.run([sys.executable,str(root/'src'/script),*args],check=True,cwd=root)
subprocess.run([sys.executable,str(root/'verify_local.py')],check=True,cwd=root)
