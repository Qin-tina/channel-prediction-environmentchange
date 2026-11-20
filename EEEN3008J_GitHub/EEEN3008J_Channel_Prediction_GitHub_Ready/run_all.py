#!/usr/bin/env python3
"""Run C then D against the supplied A/B files; never change A or B source code.

Running C overwrites its generated results, and running D overwrites D generated
results. Original A/B input files and the saved B model are read-only.
"""
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parent
steps = [
    ([sys.executable, str(ROOT/'ABC'/'03_Updated_C'/'run_C_complete.py')], ROOT),
    ([sys.executable, str(ROOT/'D'/'part_d.py'), '--abc-root', str(ROOT/'ABC'), '--out', str(ROOT/'D')], ROOT),
]
for cmd, cwd in steps:
    print('\nRunning:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)
print('\nComplete: see ABC/03_Updated_C/results and D/results + D/figures')
