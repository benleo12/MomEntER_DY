"""Build a pipeline-ready prior directory from the showered POWHEG CSV.
   python powheg_build_prior.py hooks_wgt.csv sherpa_prior_13TeV_powheg"""
import sys, os, pandas as pd, numpy as np
src, P = sys.argv[1], sys.argv[2]
os.makedirs(P + '/variations', exist_ok=True)
df = pd.read_csv(src)
for col, fn in [('qT','pT_values'), ('m','m_values'), ('dphi','dphi_values'), ('w','pT_weight')]:
    df[col].to_csv(f'{P}/{fn}.csv.gz', index=False, header=False, compression='gzip')
VAR = {'w1002':'MUR_0.5__MUF_0.5', 'w1003':'MUR_0.5__MUF_1', 'w1004':'MUR_1__MUF_0.5',
       'w1005':'MUR_1__MUF_2',   'w1006':'MUR_2__MUF_1',   'w1007':'MUR_2__MUF_2'}
for c, n in VAR.items():
    df[c].to_csv(f'{P}/variations/{n}.csv.gz', index=False, header=False, compression='gzip')
print(f"{len(df):,} events -> {P}  (<rT>={np.average(df.qT/df.m):.4f}, "
      f"{100*np.mean(df.w<0):.2f}% negative weights)")
