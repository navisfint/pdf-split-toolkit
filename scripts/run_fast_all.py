# -*- coding: utf-8 -*-
"""快速版五份串行跑 + 计时"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'C:/Users/admin/.workbuddy/binaries/python/envs/default/Scripts/python.exe'
OUTF = r'F:/workBuddy/PDF智能拆分/output_fast'
tot = time.time()
for tag in ['51','60','66','67','113']:
    out = os.path.join(OUTF, f'DHGS-GL5\u20221\u20221\u2022TJ6\u20225-{tag}')
    log = os.path.join(HERE, '_probe', f'fast_{tag}_run.log')
    t0 = time.time()
    r = subprocess.run([PY, 'split_fast.py', tag, '--out', out],
                       cwd=HERE, capture_output=True)
    dt = time.time() - t0
    print(f'{tag}: {dt:.0f}s  exit={r.returncode}', flush=True)
with open(os.path.join(HERE,'_probe','fast_all_time.txt'),'w') as f:
    f.write(f'{time.time()-tot:.0f}')
print(f'TOTAL {time.time()-tot:.0f}s')
