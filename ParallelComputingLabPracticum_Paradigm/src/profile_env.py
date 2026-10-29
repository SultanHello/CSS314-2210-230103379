import argparse, os, platform, subprocess, sys, re
from common import save
from salt import get_student_params

def query(key):
    p=subprocess.run(['sysctl','-n',key],capture_output=True,text=True)
    if p.returncode: return None
    s=p.stdout.strip()
    try: return int(s)
    except ValueError: return s

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--student-id',required=True); args=ap.parse_args()
    profile={'os':platform.platform(),'kernel':platform.release(),'architecture':platform.machine(),
             'python':sys.version,'python_implementation':platform.python_implementation(),
             'logical_threads':os.cpu_count(),'cpu_model':None,'microarchitecture':None,
             'physical_cores':None,'base_frequency_hz':None,'cache_line_bytes':None,
             'l1_data_bytes':None,'l2_bytes':None,'l3_bytes':None,'notes':[]}
    if sys.platform=='darwin':
        keys={'cpu_model':'machdep.cpu.brand_string','physical_cores':'hw.physicalcpu',
              'logical_threads':'hw.logicalcpu','base_frequency_hz':'hw.cpufrequency',
              'cache_line_bytes':'hw.cachelinesize','l1_data_bytes':'hw.l1dcachesize',
              'l2_bytes':'hw.l2cachesize','l3_bytes':'hw.l3cachesize'}
        for name,key in keys.items():
            value=query(key)
            if value is not None: profile[name]=value
        p=subprocess.run(['system_profiler','SPHardwareDataType'],capture_output=True,text=True)
        for line in p.stdout.splitlines():
            if line.strip().startswith('Chip:'): profile['cpu_model']=line.split(':',1)[1].strip()
            if 'Total Number of Cores:' in line:
                v=line.split(':',1)[1].strip(); profile['core_description']=v
                profile['physical_cores']=int(v.split()[0])
        profile['notes'].append('Unavailable sysctl fields are null; no cache sizes or clock rates are guessed. No hardware identifiers are collected.')
    else:
        profile['notes'].append('Automatic detailed topology implemented for macOS; populate missing fields from authenticated OS telemetry on other systems.')
    profile['student_id_length']=len(args.student_id)
    if len(args.student_id)!=8: profile['notes'].append('PDF requests 8 digits; user supplied ID preserved exactly, pending instructor confirmation.')
    save('host_profile.json',profile); save('salted_params.json',get_student_params(args.student_id))
    print('Saved host profile and salted parameters',flush=True)
if __name__=='__main__': main()
