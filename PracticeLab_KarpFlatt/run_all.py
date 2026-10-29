"""Run the prescribed workloads sequentially and retain raw measurements."""
import pathlib
import subprocess
import sys

if __name__ == '__main__':
    root = pathlib.Path(__file__).resolve().parent
    for filename, result in [('exp_a_serial.py', 'a.json'),
                             ('exp_b_contention.py', 'b.json'),
                             ('exp_c_fixed.py', 'c.json')]:
        subprocess.run([sys.executable, str(root / filename), '--repeats', '3',
                        '--json', str(root / 'measurements' / result)], check=True)
