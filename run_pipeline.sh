#!/bin/bash
# Fully automated MaxEnt select+run pipeline for ANY prior (publishable driver).
#   ./run_pipeline.sh <ENERGY>     e.g. 13p6TeV_v2 | 13TeV_v3 | 13TeV_pwgP
# ONE option, chosen by the user: GATE=1 (default) hands the tail back to the prior over
# [GATE_LO,GATE_HI]; GATE=0 exports the pure reweighting on every event.  Everything else is fixed.
# SELECTOR CONFIGURATION (identical for every prior): stability prune with the stat(+)scale penalty
# (SIG_MODE=statscale), ABSOLUTE floor MIN_EFF_EVENTS effective events (prior-agnostic),
# at most 15 prune rounds, stability evaluated on ALL events, fits on the 2M subsample. Fits/exports use SIG_MODE=stat.
# CONTRACT: no per-prior branches, no per-run knobs. Every sample size below means
# "all events of the prior" (NEV=2e8 clamps to the sample size) except the dual FITS,
# which are always on the fixed 2M subsample. Identical settings for every prior.
# Chain: candidate pools (tau2.0 = moments with theory uncertainty < 2%; ALL = every tabulated moment)
#        -> per-prior stability prune -> physical-sigma fit -> PROG validation plots
#        -> meta-select winner by out-of-sample distribution agreement (fit sees only
#           moments; distributions are never touched by lambda -> honest model selection)
#        -> winner: lambda_export + 29-scheme variations + per-event weights + apply-check.
set -u
E=$1; MOM="moments_${E}"; PRIOR="sherpa_prior_${E}"
export SIG_MODE=${SIG_MODE:-stat}; export FULLFIT_EXPORT=${FULLFIT_EXPORT:-1}; export MIN_EFF_EVENTS=${MIN_EFF_EVENTS:-1000000}
export GATE=${GATE:-1}; export GATE_LO=${GATE_LO:-120}; export GATE_HI=${GATE_HI:-200}
# NOTE: env assignments must be LITERAL on the command line -- bash does not re-parse a variable
# expansion as an assignment prefix, it runs it as a command.  So export them instead.
if [ "$GATE" = "1" ]; then UNG=0; export GATE_PRO=1
else                       UNG=1; unset GATE_PRO || true; fi
export UNGATED=$UNG
echo "SIG_MODE=$SIG_MODE MIN_EFF_EVENTS=$MIN_EFF_EVENTS GATE=$GATE (UNGATED=$UNG)"
SCR=${SCR:-./output}   # scratch/log directory (override with SCR=...)
mkdir -p "$SCR"; cd "$(dirname "$0")"
echo "===== PIPELINE $E  ($(date)) ====="

echo "--- [1/5] candidate pools (tau screen + S/N) ---"
if [ -f "$MOM/cand_SN.json" ] && [ -f "$MOM/cand_tau2.0.json" ]; then echo "  (pools exist, skip)"; else
ENERGY=$E NEV=20000000 PYTHONUNBUFFERED=1 python build_tau_sets.py > "$SCR/pool_$E.log" 2>&1 \
  || { echo "POOL BUILD FAILED"; tail -5 "$SCR/pool_$E.log"; exit 1; }
fi
grep -E "tau2.0|SN" "$SCR/pool_$E.log"
# ALL = every tabulated moment (the candidate list is the theory table itself; the statistical admission
# below decides what THIS prior can carry).  Built once, prior-independent.
python - "$MOM" <<'PY'
import csv,json,sys,os
mom=sys.argv[1]; out=f'{mom}/cand_ALL_raw.json'
if os.path.exists(out): print(f"  ALL: {len(json.load(open(out))['selected_moments'])} moments (exists)")
else:
    rows=[r for r in csv.DictReader(open(f'{mom}/DYMoments_N4LLp+N3LO.csv')) if r['ScaleFO']=='CV->FO' and r['ScaleRes']=='CV->Res']
    names=[]
    for r in rows:
        fs=[s for s in (r['O1'],r['O2']) if int(s.split('^')[1])>0]
        if not fs: continue
        rtp=[s for s in fs if s.split('^')[0] in ('rt','lnrt')]; dp=[s for s in fs if s.split('^')[0] in ('dphi','lndphi')]
        nm=('*'.join(rtp) if rtp else 'const^0')+'\u00d7'+('*'.join(dp) if dp else 'const^0')
        if nm not in names: names.append(nm)
    json.dump({'selected_moments':names,'n_selected':len(names),'source':'ALL (every tabulated moment; admission decides)'},open(out,'w'),indent=1,ensure_ascii=False)
    json.dump({'selected_moments':names,'n_selected':len(names),'source':'ALL'},open(f'{mom}/cand_ALL.json','w'),indent=1,ensure_ascii=False)
    print(f"  ALL: {len(names)} moments")
PY

POOLS="tau2.0"
# Statistical ADMISSION of every pool BEFORE any fit (admit_prune.py; no tolerance, no knob).  A moment
# enters only if THIS prior can carry the constraint: (1) the sample's own MC error on the moment is
# below the theory stat error the fit holds it to (tail-carried moments fail: the optimiser otherwise
# steers a handful of extreme events and descends the signed-measure chute), and (2) what the moment
# adds beyond the already-admitted ones has a target known better than the prior's spread along that
# direction (exact and near-null families fail: lambda along them is set by theory noise -- the same
# 41 moments fitted at 13 and 13.6 TeV gave lambda corr 0.62 and an oscillating dphi).  The fits then
# add the sample's MC error to the stat penalty in quadrature (prior_mc.json).  A failed admission is
# a hard stop: the raw pool must never reach the optimiser.
export ADMIT_H=${ADMIT_H:-0.1}
for P in $POOLS; do
  [ -f "$MOM/cand_${P}_raw.json" ] || cp "$MOM/cand_$P.json" "$MOM/cand_${P}_raw.json"
  AKEY="$(md5sum admit_prune.py | cut -c1-12) H=$ADMIT_H pool=$(md5sum $MOM/cand_${P}_raw.json | cut -c1-12) NEV=200000000 FIT=5000000"
  [ "$(cat $MOM/cand_${P}.admit 2>/dev/null)" = "$AKEY" ] && { echo "  [pool $P] (admitted, skip)"; continue; }
  NEV=200000000 FIT_NEV=5000000 python admit_prune.py "$PRIOR" "$MOM/DYMoments_N4LLp+N3LO.csv" "$MOM/cand_${P}_raw.json" "$MOM/cand_$P.json" > "$SCR/admit_${E}_$P.log" 2>&1 \
    || { echo "  [pool $P] ADMISSION FAILED"; tail -3 "$SCR/admit_${E}_$P.log"; exit 1; }
  grep -E '^admit: (kept|signed)' "$SCR/admit_${E}_$P.log" | sed "s/^/  [pool $P] /"; echo "$AKEY" > "$MOM/cand_${P}.admit"
done

echo "--- [2/5] stability prune + [3/5] PROG validation per pool: $POOLS ---"
for P in $POOLS; do
  echo "  [pool $P] prune ..."
  # A pool that was found NOT STABILIZABLE is recorded in a marker so a relaunch does not spend
  # hours re-proving it (POWHEG S/N: 5.5 h, re-run from scratch on resume, 2026-08-29).
  if [ -f "$MOM/unstable_$P.marker" ]; then echo "  [pool $P] NOT STABILIZABLE (recorded $(cat $MOM/unstable_$P.marker)) -> excluded, skip"; continue; fi
  if [ -f "$MOM/stable_$P.json" ]; then echo "  [pool $P] (stable set exists, skip prune)"; else
  ENERGY=$E NEV=200000000 FIT_NEV=5000000 BATCH=2000000 FRAC=0.5 MAX_STEPS=2000 MAXDROP=15 SIG_MODE=statscale \
    PYTHONUNBUFFERED=1 python select_stable.py "$MOM/cand_$P.json" "$MOM/stable_$P.json" > "$SCR/sel_${E}_$P.log" 2>&1 \
    || { rc=$?; if [ $rc -eq 3 ]; then echo "  [pool $P] NOT STABILIZABLE on this prior -> excluded from meta-selection"; date +%F_%H:%M > "$MOM/unstable_$P.marker";
           grep -E "WARNING|NOT STABILIZABLE" "$SCR/sel_${E}_$P.log" | sed "s/^/  [pool $P] /";
         else echo "  [pool $P] PRUNE FAILED (rc=$rc)"; tail -3 "$SCR/sel_${E}_$P.log"; fi; continue; }
  fi
  grep -E "STABLE at" "$SCR/sel_${E}_$P.log" 2>/dev/null | sed "s/^/  [pool $P]/"
  echo "  [pool $P] PROG validation ..."
  if [ "$(grep -c 'trusted median' "$SCR/prog_${E}_$P.log" 2>/dev/null)" = "3" ]; then echo "  [pool $P] (PROG exists, skip)"; else
  # PROG is an OUT-OF-SAMPLE AGREEMENT measure whose only job is to RANK the pools, so it must
  # run UNGATED whatever the export choice is.  With GATE_PRO=1 final_plots_pro.py takes the
  # gate branch and sys.exit()s before the three "trusted median" lines that [4/5] parses.
  # There are TWO such branches, keyed on DIFFERENT variables -- GATE_PRO at
  # final_plots_pro.py:462 and GATE at :562 -- so clearing only one still exits early.
  # Every pool then scores invalid and meta-select
  # aborts with "no pool produced a valid PROG run".  That is why GATE=0 (POWHEG) never hit
  # this and GATE=1 (Sherpa) always does.  These literal prefixes override the exports.
  ENERGY=$E NEV=200000000 FIT_NEV=5000000 NEV_BAND=10000000 BATCH=2000000 SCHMAX=28 RCOND=1e-3 \
    GATE=0 GATE_PRO=0 UNGATED=1 \
    OUT_TAG=PROG_$P \
    PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_$P.json" > "$SCR/prog_${E}_$P.log" 2>&1 \
    || { echo "  [pool $P] PROG FAILED"; tail -3 "$SCR/prog_${E}_$P.log"; continue; }
  fi
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
    meds=[float(x) for x in re.findall(r'\| trusted median=([0-9.]+)%',txt)] or [float(x) for x in re.findall(r'rew/thy median=([0-9.]+)%',txt)]
    pri=[float(x) for x in re.findall(r'prior trusted median=([0-9.]+)%',txt)]
    if len(meds)!=3: continue
    # never worse than the prior on ANY validation distribution: a set that degrades one is not a
    # deliverable whatever its N_eff (13.6 TeV UNION-47: dphi 5.4% vs prior 3.5%, gate PASS, 2026-08-29)
    if len(pri)==3 and any(m>p for m,p in zip(meds,pri)):
        print(f"  {p:8s} rT/dphi/pT = {meds[0]:.2f}/{meds[1]:.2f}/{meds[2]:.2f}  vs prior {pri[0]:.2f}/{pri[1]:.2f}/{pri[2]:.2f}  -> WORSE THAN PRIOR, refused",file=sys.stderr); continue
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
# The full-sample fit can fail to converge on a set the 2M-event prune accepted: one moment stays
# off by O(10%) while the others sit at 1e-7 (13.6 TeV tau2.0, 2026-08-28) -- a flat direction that
# only opens up on the full sample.  The export then refuses (rc=3) and names the offender in
# $MOM/fullfit_unconverged.json.  Same remedy as the prune: drop it, refit the subsample (so the
# full fit has a validated warm start again), export again.  Bounded at 3 shrinks, every step logged.
SETJ="$MOM/stable_WINNER.json"; rm -f "$MOM/fullfit_unconverged.json"
for TRY in 0 1 2 3; do
  ENERGY=$E NEV=200000000 FIT_NEV=5000000 FULLFIT=$FULLFIT_EXPORT BATCH=2000000 UNGATED=$UNG EXPORT=1 \
    PYTHONUNBUFFERED=1 python final_plots_pro.py "$SETJ" > "$SCR/exp_$E.log" 2>&1; RC=$?
  [ $RC -eq 0 ] && break
  if [ $RC -eq 3 ] && [ -f "$MOM/fullfit_unconverged.json" ]; then
    grep -E "stationarity|resid/scale|NOT CONVERGED" "$SCR/exp_$E.log" | sed 's/^/  /'
    [ $TRY -eq 3 ] && { echo "EXPORT NEVER CONVERGED after 3 shrinks"; exit 3; }
    python - "$SETJ" "$MOM/fullfit_unconverged.json" "$MOM/stable_WINNER_S.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); u=json.load(open(sys.argv[2])); w=u['worst'][0]
s['selected_moments']=[m for m in s['selected_moments'] if m!=w]; s['n_selected']=len(s['selected_moments'])
s['source']=s.get('source','')+f' | FULLFIT-shrink: dropped {w} (resid {u["resid"][0]:.3g})'
json.dump(s,open(sys.argv[3],'w'),indent=2,ensure_ascii=False); print(f"  shrink: dropped {w} -> K={s['n_selected']}")
PY
    rm -f "$MOM/fullfit_unconverged.json"; SETJ="$MOM/stable_WINNER_S.json"; cp "$SETJ" "$MOM/stable_WINNER.json"
    echo "  shrink: subsample refit of the reduced set (warm start for the full fit) ..."
    ENERGY=$E NEV=20000000 FIT_NEV=5000000 NEV_BAND=2000000 BATCH=2000000 SCHMAX=2 RCOND=1e-3 GATE=0 GATE_PRO=0 UNGATED=1 OUT_TAG=SHRINK$TRY \
      PYTHONUNBUFFERED=1 python final_plots_pro.py "$SETJ" > "$SCR/shrink_${E}_$TRY.log" 2>&1 \
      || { echo "  shrink: subsample refit FAILED (rc=$?)"; tail -3 "$SCR/shrink_${E}_$TRY.log"; exit 3; }
  else echo "EXPORT FAILED rc=$RC"; tail -3 "$SCR/exp_$E.log"; exit 1; fi
done
grep -E "wrote|log_norm" "$SCR/exp_$E.log" | tail -2
ENERGY=$E NEV=200000000 FIT_NEV=5000000 FULLFIT=$FULLFIT_EXPORT NEV_BAND=2000000 BATCH=2000000 SCHMAX=28 RCOND=1e-3 UNGATED=$UNG EXPORT_VARS=1 \
  PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_WINNER.json" > "$SCR/expv_$E.log" 2>&1
grep -E "wrote" "$SCR/expv_$E.log" | tail -1

echo "--- [5b/6] health gate: dCmax (scale-band lever), small-qT divergence, N_eff floor ---"
# v2, not v1: max|lambda| and |C| are NOT invariant (rescale a moment and they change while the
# physics does not), and v1 skipped dCmax -- the one criterion with measured discriminating power --
# whenever they tripped.  exit 2 = "cannot verify" (missing health block) must trigger a re-export,
# NEVER a moment-set substitution.
python fit_health_v2.py "$MOM" && python fit_health_v2.py "$MOM" --vars; HRC=$?
if [ $HRC -eq 2 ]; then echo "  GATE CANNOT VERIFY (rc=2) -- the export lacks its health block; aborting rather than substituting."; exit 2; fi
if [ $HRC -ne 0 ]; then
  cp "$MOM/stable_WINNER.json" "$MOM/stable_WINNER_pregate.json"
  LASTK=-1; HEALTHY=0
  # ---- 1st remedy: health-shrink.  A small-qT divergence is carried by the leading pure-lnrt tower
  # (check_smallqt.leading_tower); drop those members and keep everything else, then prune / refit /
  # export / re-gate exactly as for any set.  Bounded at 3 rounds.  Only if that cannot pass do we
  # rebuild from the pool (grow), which is far blunter (13 TeV 2026-08-29: K=41 -> K=11).
  echo "  GATE FAIL -> remedy 1: health-shrink (drop the leading lnrt tower, keep the set)"
  HSET="$MOM/stable_WINNER.json"
  for HR in 1 2 3; do
    python health_shrink.py "$MOM/lambda_export.json" "$HSET" "$MOM/stable_WINNER_H.json" || break
    ENERGY=$E NEV=200000000 FIT_NEV=5000000 BATCH=2000000 FRAC=0.5 MAX_STEPS=2000 MAXDROP=15 SIG_MODE=statscale \
      PYTHONUNBUFFERED=1 python select_stable.py "$MOM/stable_WINNER_H.json" "$MOM/stable_WINNER_HP.json" > "$SCR/sel_${E}_hshrink$HR.log" 2>&1 \
      || { echo "  health-shrink $HR: reduced set NOT STABILIZABLE"; break; }
    grep -E "STABLE at" "$SCR/sel_${E}_hshrink$HR.log" | sed "s/^/  health-shrink $HR/"
    HSET="$MOM/stable_WINNER_HP.json"
    ENERGY=$E NEV=20000000 FIT_NEV=5000000 NEV_BAND=2000000 BATCH=2000000 SCHMAX=2 RCOND=1e-3 GATE=0 GATE_PRO=0 UNGATED=1 OUT_TAG=HSHRINK$HR \
      PYTHONUNBUFFERED=1 python final_plots_pro.py "$HSET" > "$SCR/warm_${E}_hshrink$HR.log" 2>&1 \
      || { echo "  health-shrink $HR: subsample fit did not converge"; break; }
    ENERGY=$E NEV=200000000 FIT_NEV=5000000 FULLFIT=$FULLFIT_EXPORT BATCH=2000000 UNGATED=$UNG EXPORT=1 \
      PYTHONUNBUFFERED=1 python final_plots_pro.py "$HSET" > "$SCR/exp_${E}_hshrink$HR.log" 2>&1 \
      || { echo "  health-shrink $HR: export failed"; grep -E 'NOT CONVERGED|resid/scale' "$SCR/exp_${E}_hshrink$HR.log" | head -3; break; }
    python fit_health_v2.py "$MOM" || continue
    ENERGY=$E NEV=200000000 FIT_NEV=5000000 FULLFIT=$FULLFIT_EXPORT NEV_BAND=2000000 BATCH=2000000 SCHMAX=28 RCOND=1e-3 UNGATED=$UNG EXPORT_VARS=1 \
      PYTHONUNBUFFERED=1 python final_plots_pro.py "$HSET" > "$SCR/expv_${E}_hshrink$HR.log" 2>&1 || continue
    python fit_health_v2.py "$MOM" --vars && { K=$(python -c "import json;print(len(json.load(open('$HSET'))['selected_moments']))");
      echo "  GATE PASS after health-shrink round $HR: K=$K"; cp "$HSET" "$MOM/stable_WINNER.json"; HEALTHY=1; break; }
  done
  [ $HEALTHY -eq 1 ] || echo "  health-shrink did not pass -> remedy 2: rebuild the selection from the screened pool (grow)"
  [ $HEALTHY -eq 1 ] || \
  for TOL in 0.35 0.45 0.55 0.65 0.75; do
    python grow_select.py "$PRIOR" "$MOM/stable_WINNER_G.json" $TOL "$MOM/stable_WINNER_pregate.json" "$MOM/cand_ALL.json" "$MOM/cand_tau2.0.json" | tail -1
    K=$(python -c "import json;print(len(json.load(open('$MOM/stable_WINNER_G.json'))['selected_moments']))")
    echo "  grow tol=$TOL K=$K"; [ "$K" -lt 12 ] && break; [ "$K" -eq "$LASTK" ] && continue; LASTK=$K
    # A grown set comes straight from the screened pools with NO convergence prune.  Exporting it
    # directly cost a ~1 h full-sample fit per tolerance on sets that could not converge (13 TeV,
    # 2026-08-29: K=24 -> residual 8e4, cold-start |lam|=2068).  Prune it exactly like a pool, then
    # make its 2M-event warm start, THEN spend the full fit.
    ENERGY=$E NEV=200000000 FIT_NEV=5000000 BATCH=2000000 FRAC=0.5 MAX_STEPS=2000 MAXDROP=15 SIG_MODE=statscale \
      PYTHONUNBUFFERED=1 python select_stable.py "$MOM/stable_WINNER_G.json" "$MOM/stable_WINNER_GP.json" > "$SCR/sel_${E}_grow$TOL.log" 2>&1 \
      || { echo "  grow tol=$TOL: grown set NOT STABILIZABLE -> next tolerance"; continue; }
    grep -E "STABLE at" "$SCR/sel_${E}_grow$TOL.log" | sed "s/^/  grow tol=$TOL/"
    cp "$MOM/stable_WINNER_GP.json" "$MOM/stable_WINNER_G.json"
    K=$(python -c "import json;print(len(json.load(open('$MOM/stable_WINNER_G.json'))['selected_moments']))")
    ENERGY=$E NEV=20000000 FIT_NEV=5000000 NEV_BAND=2000000 BATCH=2000000 SCHMAX=2 RCOND=1e-3 GATE=0 GATE_PRO=0 UNGATED=1 OUT_TAG=GROW$TOL \
      PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_WINNER_G.json" > "$SCR/warm_${E}_grow$TOL.log" 2>&1 \
      || { echo "  grow tol=$TOL: subsample fit of the pruned grown set did not converge -> next tolerance"; continue; }
    ENERGY=$E NEV=200000000 FIT_NEV=5000000 FULLFIT=$FULLFIT_EXPORT BATCH=2000000 UNGATED=$UNG EXPORT=1 \
      PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_WINNER_G.json" > "$SCR/exp_${E}_grow$TOL.log" 2>&1 || continue
    python fit_health_v2.py "$MOM" || continue
    ENERGY=$E NEV=200000000 FIT_NEV=5000000 FULLFIT=$FULLFIT_EXPORT NEV_BAND=2000000 BATCH=2000000 SCHMAX=28 RCOND=1e-3 UNGATED=$UNG EXPORT_VARS=1 \
      PYTHONUNBUFFERED=1 python final_plots_pro.py "$MOM/stable_WINNER_G.json" > "$SCR/expv_${E}_grow$TOL.log" 2>&1 || continue
    python fit_health_v2.py "$MOM" --vars && { echo "  GATE PASS after fallback: tol=$TOL K=$K"; cp "$MOM/stable_WINNER_G.json" "$MOM/stable_WINNER.json"; HEALTHY=1; break; }
  done
  [ $HEALTHY -eq 1 ] || { echo "GATE NEVER PASSED"; exit 1; }
else
  echo "  GATE PASS"
fi
ENERGY=$E NEV=200000000 FIT_NEV=5000000 FULLFIT=$FULLFIT_EXPORT BATCH=2000000 UNGATED=$UNG GATE=$GATE GATE_LO=$GATE_LO GATE_HI=$GATE_HI GATE_PRO=0 \
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
echo "--- [6/6] validate the EXPORTED deliverable (not a refit): shipped weights vs the calculation ---"
# The per-pool validation above fits on 2M; the export ships a full-sample fit, and a gate fallback
# can replace the set entirely.  So the deliverable itself is histogrammed here -- no refit, no cache.
ENERGY=$E python validate_shipped_weights.py "$MOM/sherpa_${E}_maxent_weights.npz" 2>&1 | tee "$SCR/valship_$E.log" | sed "s/^/  /"
echo "===== PIPELINE $E DONE ($(date)) ====="
