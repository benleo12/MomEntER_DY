"""Objective fit-health gate. exit 0 = healthy, 1 = sick.
   python fit_health.py <moments_dir> [--vars]"""
import sys, json, numpy as np
M=sys.argv[1]; CHECK_VARS='--vars' in sys.argv
LMAX,CMAX,DCMAX=30.0,5.0,0.5
d=json.load(open(f'{M}/lambda_export.json')); l=np.abs(np.array(d['lambda_physical']))
lm,c=l.max(),abs(d['log_norm_shift']); ok=(lm<LMAX and c<CMAX)
msg=f"maxlam={lm:.2f}(<{LMAX:g}) |C|={c:.2f}(<{CMAX:g})"
if CHECK_VARS and ok:
    v=json.load(open(f'{M}/lambda_export_variations.json'))['schemes']
    Cc=v['central']['log_norm_shift']
    dc=max(abs(s['log_norm_shift']-Cc) for s in v.values()); ok=dc<DCMAX
    msg+=f" dCmax={dc:.4f}(<{DCMAX:g})"
print(f"HEALTH {'PASS' if ok else 'FAIL'}: {msg}")
sys.exit(0 if ok else 1)
