#!/bin/bash
# Fully automated MaxEnt select+run pipeline for ANY prior (publishable driver).
#   ./run_pipeline.sh <ENERGY>     e.g. 13p6TeV_v2 | 13TeV_v2 | 13TeV | 13p6TeV
# Chain: candidate pools (tau2.0, S/N, union[, prior deliverable set as continuity pool])
#        -> per-prior stability prune -> physical-sigma fit -> PROG validation plots
#        -> meta-select winner by out-of-sample distribution agreement (fit sees only
#           moments; distributions are never touched by lambda -> honest model selection)
#        -> winner: lambda_export + 29-scheme variations + per-event weights + apply-check.
set -u
E=$1; MOM="moments_${E}"; PRIOR="sherpa_prior_${E}"
export SIG_MODE=${SIG_MODE:-statscale}; echo "SIG_MODE=$SIG_MODE"
SCR=${SCR:-./output}   # scratch/log directory (override with SCR=...)
mkdir -p "$SCR"; cd "$(dirname "$0")"
echo "===== PIPELINE $E  ($(date)) ====="

echo "--- [1/5] candidate pools (tau screen + S/N) ---"
ENERGY=$E NEV=20000000 PYTHONUNBUFFERED=1 python build_tau_sets.py > "$SCR/pool_$E.log" 2>&1 \
  || { echo "POOL BUILD FAILED"; tail -5 "$SCR/pool_$E.log"; exit 1; }
grep -E "tau2.0|SN" "$SCR/pool_$E.log"
python - "$MOM" <<'PY'
import json,sys
mom=sys.argv[1]
a=json.load(open(f'{mom}/cand_tau2.0.json'))['selected_moments']
b=json.load(open(f'{mom}/cand_SN.json'))['selected_moments']
u=sorted(set(a)|set(b))
json.dump({'selected_moments':u,'n_selected':len(u),'source':'UNION(tau2.0,SN)'},open(f'{mom}/cand_UNION.json','w'),indent=2)
print(f"  UNION: {len(u)} moments")
PY

POOLS="tau2.0 SN UNION"
if [ "$E" = "13TeV_v2" ]; then    # continuity pool: the shipped 13 TeV deliverable set
  cp ../ps/moments_13TeV/stable_tau2.0.json "$MOM/cand_ship52.json" 2>/dev/null || cp moments_13TeV/stable_tau2.0.json "$MOM/cand_ship52.json"
  POOLS="$POOLS ship52"
fi

echo "--- [2/5] stability prune + [3/5] PROG validation per pool: $POOLS ---"
for P in $POOLS; do
  echo "  [pool $P] prune ..."
  ENERGY=$E NEV=${NEV_PIPE:-50000000} FIT_NEV=${FIT_PIPE:-2000000} BATCH=2000000 FRAC=0.5 MAX_STEPS=200 MAXDROP=60 \
    PYTHONUNBUFFERED=1 python select_stable.py "$MOM/cand_$P.json" "$MOM/stable_$P.json" > "$SCR/sel_${E}_$P.log" 2>&1 \
    || { echo "  [pool $P] PRUNE FAILED"; tail -3 "$SCR/sel_${E}_$P.log"; continue; }
  grep -E "STABLE at" "$SCR/sel_${E}_$P.log" | sed "s/^/  [pool $P]/"
  echo "  [pool $P] PROG validation ..."
  ENERGY=$E NEV=${NEV_PIPE:-50000000} FIT_NEV=${FIT_PIPE:-2000000} NEV_BAND=10000000 BATCH=2000000 SCHMAX=28 RCOND=1e-3 \
    GATE_PRO=1 GATE_LO=120 GATE_HI=200 OUT_TAG=PROG_$P \
    PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_$P.json" > "$SCR/prog_${E}_$P.log" 2>&1 \
    || { echo "  [pool $P] PROG FAILED"; tail -3 "$SCR/prog_${E}_$P.log"; continue; }
  grep -E "rew/thy" "$SCR/prog_${E}_$P.log" | sed "s/^/  [pool $P]/"
done

echo "--- [4/5] meta-select winner (mean of median dist deviations) ---"
WINNER=$(python - "$E" "$SCR" "$POOLS" <<'PY'
import re,sys,json,shutil
E,scr,pools=sys.argv[1],sys.argv[2],sys.argv[3].split()
best=None
for p in pools:
    try: txt=open(f'{scr}/prog_{E}_{p}.log').read()
    except FileNotFoundError: continue
    meds=[float(x) for x in re.findall(r'trusted median=([0-9.]+)%',txt)] or [float(x) for x in re.findall(r'rew/thy median=([0-9.]+)%',txt)]
    if len(meds)!=3: continue
    score=sum(meds)/3
    print(f"  {p:8s} rT/dphi/pT = {meds[0]:.2f}/{meds[1]:.2f}/{meds[2]:.2f}  score={score:.2f}",file=sys.stderr)
    if best is None or score<best[1]: best=(p,score)
assert best, "no pool produced a valid PROG run"
mom=f"moments_{E}"
shutil.copy(f'{mom}/stable_{best[0]}.json',f'{mom}/stable_WINNER.json')
json.dump({'winner':best[0],'score':best[1]},open(f'{mom}/WINNER.json','w'))
print(best[0])
PY
) || { echo "META-SELECT FAILED"; exit 1; }
echo "  WINNER: $WINNER  (set -> $MOM/stable_WINNER.json)"

echo "--- [5/5] winner exports + apply-check ---"
ENERGY=$E NEV=${NEV_FULL:-51200000} FIT_NEV=${FIT_PIPE:-2000000} BATCH=2000000 EXPORT=1 \
  PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_WINNER.json" > "$SCR/exp_$E.log" 2>&1
grep -E "wrote|log_norm" "$SCR/exp_$E.log" | tail -2
ENERGY=$E NEV=${NEV_FULL:-51200000} FIT_NEV=${FIT_PIPE:-2000000} NEV_BAND=2000000 BATCH=2000000 SCHMAX=28 RCOND=1e-3 EXPORT_VARS=1 \
  PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_WINNER.json" > "$SCR/expv_$E.log" 2>&1
grep -E "wrote" "$SCR/expv_$E.log" | tail -1
ENERGY=$E NEV=${NEV_FULL:-51200000} FIT_NEV=${FIT_PIPE:-2000000} BATCH=2000000 GATE=1 GATE_LO=120 GATE_HI=200 \
  WRITE_WEIGHTS="$MOM/sherpa_${E}_maxent_weights" \
  PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_WINNER.json" > "$SCR/wts_$E.log" 2>&1
grep -E "wrote" "$SCR/wts_$E.log" | tail -1
python - "$E" <<'PY' 2>/dev/null
import numpy as np, sys, optimizer_DY_unc as o
from apply_lambdas import reweight
E=sys.argv[1]; mom=f"moments_{E}"
p=o.load_prior(f"sherpa_prior_{E}"); N=2_000_000
rt=p['rT'][:N].astype(float); dd=p['d'][:N].astype(float); pT=p['pT'][:N].astype(float); w0=p['w'][:N].astype(float)
m=pT/np.maximum(rt,1e-12)
wa=reweight(w0,pT,m,dd,jpath=f'{mom}/lambda_export.json')
ref=np.load(f'{mom}/sherpa_{E}_maxent_weights.npz')['weights'][:N].astype(float)
bulk=(pT<120)&(np.abs(w0)>0)&(np.abs(wa)>0)
r=ref[bulk]/wa[bulk]; c=np.median(r)
rel=np.abs(ref[bulk]-c*wa[bulk])/np.maximum(np.abs(ref[bulk]),1e-30)
print("  APPLY-CHECK %s: median|rel|=%.2e p99=%.2e  %s"%(E,np.median(rel),np.percentile(rel,99),"OK" if np.median(rel)<1e-6 else "FAIL"))
PY
echo "===== PIPELINE $E DONE ($(date)) ====="
