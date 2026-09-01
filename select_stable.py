"""Automated stability selection (prior-agnostic). Starting from a candidate moment set,
iteratively DROP the moment that contributes most to the worst reweight-factor event at the
apply scale, until N_eff is preserved (>= FRAC * prior N_eff). Only ever fits near-full,
well-conditioned sets, so it avoids the ill-conditioned-subset thrashing of greedy accumulate.
Same feature/standardization/physical-sigma conventions as final_plots_pro (the run).
Usage: python select_stable.py <candidate.json> <out.json>
Env: ENERGY(13TeV|13p6TeV) NEV FIT_NEV BATCH FRAC(0.5) SIG_FLOOR_REL(0.005) MAXDROP(15)
"""
import os,sys,json,io,contextlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"4")
import numpy as np, optimizer_DY_unc as o
SRC=sys.argv[1]; OUT=sys.argv[2]
NEV=int(os.environ.get('NEV','50000000')); FIT=int(os.environ.get('FIT_NEV','2000000'))
B_=int(os.environ.get('BATCH','2000000')); FRAC=float(os.environ.get('FRAC','0.5'))
SIGREL=float(os.environ.get('SIG_FLOOR_REL','0.005')); MAXDROP=int(os.environ.get('MAXDROP','15'))
ACC="N4LL'+N3LO"; ACC_SLUG="N4LLp+N3LO"
ENE=os.environ.get('ENERGY','13TeV'); _M={'13TeV':'13TeV','13p6TeV':'13p6TeV','13TeV_v2':'13TeV_v2','13p6TeV_v2':'13p6TeV_v2','13TeV_m':'13TeV_m','13TeV_py':'13TeV_py','13TeV_pwg':'13TeV_pwg','13TeV_pwg10M':'13TeV_pwg10M','13TeV_pwg10M_M0505':'13TeV_pwg10M_M0505','13TeV_v3_M0505':'13TeV_v3_M0505','13TeV_pwg10M_M051':'13TeV_pwg10M_M051','13TeV_v3_M051':'13TeV_v3_M051','13TeV_pwg10M_M105':'13TeV_pwg10M_M105','13TeV_v3_M105':'13TeV_v3_M105','13TeV_pwg10M_M12':'13TeV_pwg10M_M12','13TeV_v3_M12':'13TeV_v3_M12','13TeV_pwg10M_M21':'13TeV_pwg10M_M21','13TeV_v3_M21':'13TeV_v3_M21','13TeV_pwg10M_M22':'13TeV_pwg10M_M22','13TeV_v3_M22':'13TeV_v3_M22','13TeV_lomlm':'13TeV_lomlm','13TeV_v3':'13TeV_v3'}
# any tag with a prior directory on disk is accepted; typos still fail loudly
if ENE not in _M and os.path.isdir(f"sherpa_prior_{ENE}"): _M[ENE]=ENE
_M=_M[ENE]
MOM=f"moments_{_M}"; PRIOR=f"sherpa_prior_{_M}"; CSV=f"{MOM}/DYMoments_{ACC_SLUG}.csv"
print(f"  ENERGY={ENE} prior={PRIOR} MOM={MOM}")
names=json.load(open(SRC)).get('selected_moments') or []
def fampow(s): f,k=s.split('^'); return (f,int(k))
def parse_part(p): return [] if p=='const^0' else [fampow(s) for s in p.split('*')]
def facj1(f,k,x): return np.ones(len(x)) if k==0 else (x**k if f in('rt','dphi') else np.log(np.maximum(x,1e-12))**k)
def side(fl,x):
    out=np.ones(len(x))
    for f,k in fl: out=out*facj1(f,k,x)
    return out
# targets + physical sigma
mom=o.load_moments(CSV); mbp={(a,b):v for a,b,v,u in mom}; mbu={(a,b):u for a,b,v,u in mom}
# CORRELATED selection: prune against the SAME fixed-order variation of the targets that the
# prior was varied with, so the surviving set is the one that is well determined for THAT pair.
_TS=os.environ.get('TARGET_SCHEME','')
if _TS:
    _ms=o.load_moments_for_scale(CSV,f"{_TS}->FO","CV->Res")
    _mbv={(a,b):v for a,b,v,u in _ms}
    _n=sum(1 for k in mbp if k in _mbv and _mbv[k]!=mbp[k])
    mbp={**mbp,**_mbv}
    print(f"  TARGET_SCHEME={_TS}: selecting against {_TS}->FO targets ({_n} moments differ)")
ssc=o.compute_sigma_theory(CSV)
SIG_MODE=os.environ.get('SIG_MODE','statscale')
if SIG_MODE in ('stat','cov'): ssc={}; print(f"  SIG_MODE={SIG_MODE} (diag penalty = stat only)")
# Prior-side MC error of every moment (admit_prune.py, full sample) is added in quadrature to the
# theory sigma: both are statistical.  A constraint cannot be held tighter than the sample resolves it.
_MC=json.load(open(f'{MOM}/prior_mc.json')) if os.path.exists(f'{MOM}/prior_mc.json') else {}
if _MC: print(f"  penalty sigma = theory sigma (+) prior MC error (prior_mc.json: {len(_MC)-1} moments)")
_MCWARNED=set()
def lookup(dic,allf):
    fs=[f'{f}^{k}' for f,k in allf]
    if len(fs)==1:
        for z in ('dphi^0','lndphi^0','rt^0','lnrt^0'):
            if (fs[0],z) in dic: return dic[(fs[0],z)]
            if (z,fs[0]) in dic: return dic[(z,fs[0])]
    elif len(fs)==2:
        if (fs[0],fs[1]) in dic: return dic[(fs[0],fs[1])]
        if (fs[1],fs[0]) in dic: return dic[(fs[1],fs[0])]
    return None
def sig_for(allf):
    fs=[f'{f}^{k}' for f,k in allf]; cs=[]
    if len(fs)==1:
        for z in ('dphi^0','lndphi^0','rt^0','lnrt^0'): cs+=[(fs[0],z),(z,fs[0])]
    elif len(fs)==2: cs+=[(fs[0],fs[1]),(fs[1],fs[0])]
    for k in cs:
        sc=ssc.get(k); st=mbu.get(k)
        if sc is not None or st is not None: return ((sc or 0)**2+(st or 0)**2)**0.5
    return None
# data
p=o.load_prior(PRIOR); Nf=len(p['rT']); NEV=min(NEV,Nf); FIT=min(FIT,NEV)
ns=min(500000,Nf); rt_ref=p['rT'][:ns].astype(float); d_ref=p['d'][:ns].astype(float)   # pinned std ref
idx=np.sort(np.random.default_rng(42).choice(Nf,NEV,replace=False))
rt=p['rT'][idx].astype(float); d=p['d'][idx].astype(float); w=p['w'][idx].astype(float); pT=p['pT'][idx].astype(float); del p
GLO=float(os.environ.get('GATE_LO',120.)); GHI=float(os.environ.get('GATE_HI',200.))
_t=np.clip((pT-GLO)/(GHI-GLO),0,1); BETA=1.0-(6*_t**5-15*_t**4+10*_t**3)   # gate profile (1 bulk, 0 tail)
LOGIT_CAP=float(os.environ.get('LOGIT_CAP','500'))  # overflow guard for downstream exp()
DROP_PER_ROUND=5   # fixed: the prune removes at most 5 moments per round, blow-up or not (no one-shot wipeout)
prior_neff=w.sum()**2/np.sum(w*w)
neg_frac=float(np.mean(w<0))   # used by the thrash warning below
MIN_EFF=float(os.environ.get('MIN_EFF_EVENTS','1000000'))   # ABSOLUTE floor (effective events).
# Prior-agnostic by construction: what a prediction needs is a number of effective events, not a
# fraction of the prior's own N_eff.  The relative form (FRAC*prior_neff) punishes clean unit-weight
# priors, which start at N_eff=100%, far harder than signed-weight ones -- that asymmetry is why the
# same pool pruned to 61 moments on POWHEG and 47 on Sherpa.  FRAC is kept only as a printed diagnostic.
SEL,FAC=o.fit_subsample(rt,d,FIT); WFIT=w[SEL]*FAC   # random FIT + extreme-feature tails (importance-weighted)
print(f"  fit subsample: {len(SEL):,} events = {int((FAC!=1).sum()):,} random + {int((FAC==1).sum()):,} tail (rT range [{rt[SEL].min():.3g},{rt[SEL].max():.3g}])")
print(f"  NEV={NEV:,} FIT={FIT:,}  prior N_eff={prior_neff/NEV*100:.2f}%  FRAC={FRAC}  MIN_EFF={MIN_EFF:,.0f}")
_vk=o.get_scale_variations(CSV)[1] if SIG_MODE=='cov' else None
def build(nm):
    facs=[(parse_part(n.split('×')[0]),parse_part(n.split('×')[1])) for n in nm]
    sF=np.array([side(fa,rt_ref).std() for fa,fb in facs]); sG=np.array([side(fb,d_ref).std() for fa,fb in facs]); sF[sF==0]=1; sG[sG==0]=1
    Tc=np.array([lookup(mbp,fa+fb) or 0.0 for fa,fb in facs])
    Sig=np.array([ (sig_for(fa+fb) or SIGREL*abs(Tc[k])) for k,(fa,fb) in enumerate(facs)])
    if _MC:
        _miss=[n for n in nm if n not in _MC]
        if _miss and not _MCWARNED: _MCWARNED.add(1); print(f"  !! prior_mc.json lacks {len(_miss)} moments (sigma_MC=0 for them): {_miss[:5]}")
        Sig=np.sqrt(Sig**2+np.array([_MC.get(n,{}).get('sig_mc',0.0) for n in nm])**2)
    COV=None
    if SIG_MODE=='cov':
        def tgt_of(fo,res):
            ms=o.load_moments_for_scale(CSV,fo,res); mb={(a,b):v for a,b,v,u in ms}
            def tg(allf):
                v=lookup(mb,allf); return v
            return np.array([ (tgt if (tgt:=tg(fa+fb)) is not None else Tc[k]) for k,(fa,fb) in enumerate(facs)])
        D=np.vstack([ (tgt_of(fo,res)-Tc)/(sF*sG) for (fo,res) in _vk ])
        COV=(D.T@D)/len(D)+np.diag(np.maximum(Sig/(sF*sG),1e-12)**2)
    return facs,sF,sG,Tc,Sig,COV
import time
def worst_from(logit,ev): return None   # placeholder, replaced below per-call
def fit_apply(nm,lam0=None,sel=None,wfit=None,teps=None):
    """Fit nm on the fit subsample (SEL/WFIT unless sel/wfit are given), apply to all NEV events.
    Returns (neff, worst, ev, lam, logit, diag) -- logit over NEV and diag = Hessian eigen diagnostics
    at the solution -- so the caller can run the uniqueness test without another apply pass."""
    if sel is None: sel,wfit=SEL,WFIT
    facs,sF,sG,Tc,Sig,COV=build(nm); K=len(facs)
    def Fs(a,b): return np.column_stack([side(fa,rt[a:b]) for fa,fb in facs])/sF
    def Gs(a,b): return np.column_stack([side(fb,d[a:b]) for fa,fb in facs])/sG
    def Fsi(s): return np.column_stack([side(fa,rt[s]) for fa,fb in facs])/sF
    def Gsi(s): return np.column_stack([side(fb,d[s]) for fa,fb in facs])/sG
    pairs=np.array([(k,k) for k in range(K)],np.int64); ii=pairs[:,0]; jj=pairs[:,1]
    Ts=Tc/(sF*sG); sig=list(np.maximum(Sig/(sF*sG),1e-12))
    if teps is not None: Ts=Ts+np.asarray(teps,float)*np.asarray(sig)   # targets fluctuated within their uncertainty
    t0=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m=o.MaxEntDual(Fsi(sel),Gsi(sel),pairs,Ts,wfit,sigmas_target=sig,cov_target=COV)
        if lam0 is not None and len(lam0)==K: m.lam=lam0.copy()
        o.optimize_newton(m,max_steps=int(os.environ.get('MAX_STEPS','120')),tol=1e-9,verbose=False)
    lam=m.lam.copy(); tfit=time.time()-t0
    st=getattr(m,'fit_status',{'converged':True,'reason':'unknown','accepted_steps':-1,'max_pull':float('nan')})
    # A cold start must actually move: lam=0 with N_eff = prior N_eff is not a fit of anything.
    _ok=bool(np.all(np.isfinite(lam)) and st['converged'] and (st['accepted_steps']!=0 or lam0 is not None or st['reason']=='converged'))
    if not _ok:
        # A fit that did not converge is NOT a stable set, whatever N_eff says.  Two silent failure
        # modes used to pass as STABLE: (a) the optimiser cannot take a single step from lam=0 on an
        # ill-conditioned pool and returns lam=0, whose N_eff is the prior's and clears any floor
        # (13.6 TeV S/N and UNION pools, 2026-08-28, 102 and 125 moments 'STABLE' at iter 0);
        # (b) lam runs off along a flat direction to |lam|~1e6 with pulls ~1e9.  Neither has a
        # meaningful worst event, so the drop rule uses the Hessian instead: the moments with the
        # largest components in its near-null eigenvectors span the degenerate direction.
        _lam=lam if np.all(np.isfinite(lam)) else np.zeros(K)
        # Drop rule = the stationarity residual itself, |g_k|/max(|t_k|,sigma_k): the moment(s) the
        # optimiser cannot satisfy.  It is the quantity the export gates on and the quantity its shrink
        # loop drops by, so prune and export apply ONE rule.  (The Hessian is not usable here: on a
        # signed-weight sample the weighted covariance is indefinite -- diag min -388 measured on
        # 13 TeV tau2.0 -- so 'near-null direction' has no meaning.)  Every moment more than 100% off
        # goes, capped at DROP_PER_ROUND; at least the worst one always goes.
        try:
            _,g_,H_,_=m.dual_loss_grad_hess(_lam)
            score=np.abs(g_)/np.maximum(np.maximum(np.abs(m.targets),m.sigma),1e-300)
            _hd=float(np.min(np.diag(H_))) if np.all(np.isfinite(H_)) else float('nan')
        except Exception:
            score=np.abs(Fsi(SEL)*Gsi(SEL)).max(0); _hd=float('nan')
        if not np.all(np.isfinite(score)): score=np.where(np.isfinite(score),score,np.inf)
        order=np.argsort(-score); ndrop=max(1,min(DROP_PER_ROUND,int((score>1.0).sum())))
        wl=[int(k) for k in order[:ndrop]]; how=f'largest stationarity residual (Hessian diag min {_hd:.3g})'
        print(f"  !! FIT NOT CONVERGED ({st['reason']}, accepted steps={st['accepted_steps']}, max resid/scale={st.get('max_rel',float('nan')):.3g} (tol {st.get('tol',float('nan')):g}), max|pull|={st['max_pull']:.3g},"
              f" |lam|max={np.abs(_lam).max():.3g}): NOT a stable set regardless of N_eff.\n"
              f"     dropping by {how}: {[(nm[k],float(f'{score[k]:.3g}')) for k in wl]}",flush=True)
        print(f'      [fit {tfit:.0f}s]',flush=True)
        return 0.0, wl, (float('nan'),float('nan'),float('nan')), _lam, None, {}
    # apply at NEV: ONE feature pass -> store logits, then vectorized max-event + N_eff
    logit=np.empty(NEV)
    for a in range(0,NEV,B_):
        b=min(a+B_,NEV); logit[a:b]=(Fs(a,b)[:,ii]*Gs(a,b)[:,jj])@lam
    tapp=time.time()-t0-tfit
    ev=int(np.argmax(logit))
    # GATED deliverable weights: w_g = beta*w0*exp(logit - C) + (1-beta)*w0, with C the
    # normalization shift (log of weighted-mean factor). Stability = gated N_eff preserved
    # AND no overflow risk anywhere (|logit-C| < LOGIT_CAP) so event-level exp() is safe.
    la=np.log(np.maximum(np.abs(w),1e-300)); gmax=float((la+logit).max())
    e0=np.exp(la+logit-gmax); S1u=float((np.where(w>=0,1.,-1.)*e0).sum()); S0=float(w.sum())
    if not (S1u>0):
        # TOTAL blow-up (weights all underflow): drop ALL offending moments at once, not
        # a cap of 5.  Otherwise a pool with many high-power moments (e.g. rt^4*lnrt^4,
        # huge at large rt) collapses N_eff to 0 and the timid 5/iter drop needs many slow
        # full-sample iterations to escape.  Aggressive here, gentle (below) near stability.
        contrib=np.array([lam[k]*(side(facs[k][0],rt[ev:ev+1])[0]/sF[k])*(side(facs[k][1],d[ev:ev+1])[0]/sG[k]) for k in range(K)])
        order=np.argsort(-np.abs(contrib)); wl=[int(k) for k in order if abs(contrib[k])>5.0] or [int(order[0])]
        return 0.0, wl[:DROP_PER_ROUND], (float(logit[ev]),float(rt[ev]),float(d[ev])), lam, None, {}
    C=gmax+np.log(S1u/S0)                      # normalization shift
    if float(np.max(np.abs(logit-C)))>LOGIT_CAP:   # overflow guard fails -> unstable
        neff=0.0
    else:
        wg=BETA*w*np.exp(logit-C)+(1.0-BETA)*w
        S1=float(wg.sum()); S2=float((wg*wg).sum()); neff=(S1*S1/S2) if S2>0 else 0.0
    # per-moment contribution at the worst-logit event -> which moment destabilizes most
    contrib=np.array([lam[k]*(side(facs[k][0],rt[ev:ev+1])[0]/sF[k])*(side(facs[k][1],d[ev:ev+1])[0]/sG[k]) for k in range(K)])
    order=np.argsort(-np.abs(contrib))
    # batch-drop all moments contributing |>5| to the worst event's logit.  Cap at 5/iter
    # ONLY when the fit is stable enough to have a positive N_eff (gentle fine-tuning);
    # on a total blow-up (N_eff=0) drop them all at once so a badly-conditioned pool
    # escapes in one iteration instead of many slow full-sample passes.
    worst=[int(k) for k in order if abs(contrib[k])>5.0] or [int(order[0])]
    worst=worst[:DROP_PER_ROUND]   # ALWAYS at most DROP_PER_ROUND per round: one extreme event must never wipe a pool in one step
    # Hessian at the solution (cov + diag(sigma^2)).  On a positive-weight prior it is PSD (convex dual,
    # unique minimum); on a signed-weight prior it need not be -- measured diag min -388 on Sherpa 13 TeV.
    # A negative eigenvalue means the point is not a verified minimum of a convex problem: report it.
    try:
        _,_,H_,_=m.dual_loss_grad_hess(lam); H_=0.5*(H_+H_.T); _sc=np.sqrt(np.maximum(np.abs(np.diag(H_)),1e-300))
        _ew=np.linalg.eigvalsh(H_/np.outer(_sc,_sc)); diag={'hess_eig_min':float(_ew.min()),'hess_eig_max':float(_ew.max()),'hess_diag_min':float(np.diag(H_).min()),'C':float(C)}
    except Exception: diag={'C':float(C)}
    print(f'      [fit {tfit:.0f}s, apply {tapp:.0f}s]  Hessian(normalised) eig min={diag.get("hess_eig_min",float("nan")):+.3g} max={diag.get("hess_eig_max",float("nan")):.3g}',flush=True)
    return neff, worst, (float(logit[ev]),float(rt[ev]),float(d[ev])), lam, logit, diag
cur=list(names)
lam0=None   # iter0 cold-starts (like the run); drop-iters warm-start from previous solution
_hist=[]    # full provenance: per-iteration K, N_eff, worst event, dropped moments
_thrash=0; _prev_logit=None; _stall=[]; _stable=False
for it in range(MAXDROP+1):
    if len(cur)==0:
        print("  POOL EXHAUSTED: every candidate was pruned; no stable set from this pool."); break
    neff,worst,ev,lam,logit,hdiag=fit_apply(cur,lam0)
    # ---- thrash detector -------------------------------------------------------------
    # A rank-deficient pool makes the penalised dual unbounded below on a signed-weight
    # sample: logZ = log(sum w e^{lam.phi}) is taken over the NET measure, so directions
    # exist where positive and negative weights cancel, the net measure -> 0+ and the
    # objective -> -inf while N_eff -> 0.  The optimiser descends such a chute; dropping a
    # few moments lands it in the next one.  Symptom: N_eff pinned at ~0 with no downward
    # trend in the worst logit, one expensive refit per round.  Say so, and escalate.
    _dead = neff < 0.01*(MIN_EFF if MIN_EFF>0 else FRAC*prior_neff)
    _nogain = (not np.isfinite(ev[0])) or ((_prev_logit is not None) and np.isfinite(_prev_logit) and (abs(ev[0]) > 0.5*abs(_prev_logit)))   # a non-converged round is never 'gain'
    _thrash = _thrash+1 if (_dead and _nogain) else 0
    _prev_logit = ev[0]
    if _thrash == 2:
        _stall.append(it)
        print(f"  !! WARNING: pool is RANK-DEFICIENT on this prior.  N_eff has been ~0 for 3 rounds with no\n"
              f"     decrease in the worst logit ({ev[0]:.3g}).  The dual is descending an unbounded direction\n"
              f"     (net-measure cancellation on {100*neg_frac:.1f}% negative weights), not converging.\n"
              f"     Escalating the drop from {DROP_PER_ROUND} to {2*DROP_PER_ROUND} moments/round.", flush=True)
        DROP_PER_ROUND = 2*DROP_PER_ROUND
    if _thrash >= 5:
        print(f"  !! POOL NOT STABILIZABLE on this prior: still N_eff~0 at K={len(cur)} after escalation.\n"
              f"     Excluding this pool from the meta-selection (this is a property of the pool on THIS\n"
              f"     prior, not a failure of the run).", flush=True)
        raise SystemExit(3)
    _hist.append({'iter':it,'K':len(cur),'neff':float(neff),'max_logit':float(ev[0]),'rt':float(ev[1]),'d':float(ev[2]),'dropped':[cur[k] for k in worst] if neff<(MIN_EFF if MIN_EFF>0 else FRAC*prior_neff) else []})
    frac=neff/(FRAC*prior_neff) if prior_neff>0 else 0
    _nd=(MIN_EFF if MIN_EFF>0 else FRAC*prior_neff)
    print(f"  iter {it}: {len(cur)} moments  N_eff={neff/NEV*100:.3f}% ({neff:,.0f} eff-ev, need>={_nd:,.0f})  max-logit={ev[0]:.1f} @rt={ev[1]:.2f},d={ev[2]:.2e}")
    # Stability floor. Physically what matters is the ABSOLUTE number of effective events
    # available for predictions, not a fraction of the prior's own N_eff -- the relative
    # form punishes clean (unit-weight) priors, which start at N_eff=100%, far harder than
    # signed-weight ones. MIN_EFF>0 selects the absolute criterion.
    need = MIN_EFF if MIN_EFF>0 else FRAC*prior_neff        # absolute floor (see MIN_EFF above)
    if neff>=need:
        # Determinacy is enforced upstream by the conditioning prune of the pool (COND_TOL in
        # run_pipeline): moments whose feature column is within COND_TOL of the span of the others
        # never reach the optimiser.  A per-fit refit with independently fluctuated targets was tried
        # (2026-08-29) and withdrawn: collinear moments have ~fully correlated theory errors, so
        # independent 1-sigma shifts form a target no distribution can satisfy and every set --
        # including the convex POWHEG reference -- 'fails'.
        _hist.append({'iter':it,'K':len(cur),'hess':hdiag})
        print(f"  STABLE at {len(cur)} moments.  (N_eff={neff:,.0f} eff-events >= {need:,.0f})"); _stable=True; break
    for k in sorted(worst,reverse=True):
        print(f"    drop: {cur[k]}")
        cur.pop(k)   # cold-start every iter: warm-start is counterproductive on this collinear landscape (57s cold vs 722s warm)
if not _stable:
    # MAXDROP rounds exhausted (or pool emptied) without ever meeting the floor with a converged fit.
    # Writing the last set here would hand an UNEVALUATED set to the meta-selection -- that is how a
    # POWHEG UNION set with N_eff=0 at every round became 'stable_UNION.json' (2026-08-29).
    print(f"  !! POOL NOT STABILIZABLE within MAXDROP={MAXDROP} rounds (last K={len(cur)}): no stable set written.",flush=True)
    raise SystemExit(3)
import hashlib as _hl
_prov={'NEV':int(NEV),'FIT_NEV':int(FIT),'BATCH':int(B_),'FRAC':FRAC,'MIN_EFF':int(MIN_EFF),'MAXDROP':int(MAXDROP),
       'SIG_MODE':SIG_MODE,'SIG_FLOOR_REL':SIGREL,'TARGET_SCHEME':_TS,'DROP_PER_ROUND':DROP_PER_ROUND,
       'MAX_STEPS':int(os.environ.get('MAX_STEPS','120')),'prior':PRIOR,'targets':CSV,'candidate':SRC,'K_in':len(names),
       'prior_neff':float(prior_neff),'fit_subsample':{'n_random':int((FAC!=1).sum()),'n_tail':int((FAC==1).sum()),'q':1e-5},'code_md5':_hl.md5(open(__file__,'rb').read()).hexdigest()[:12],'history':_hist}
json.dump({'selected_moments':cur,'n_selected':len(cur),'source':f'stable @{NEV//10**6}M subset of {SRC} (auto prune)','FRAC':FRAC,'energy':ENE,'provenance':_prov},open(OUT,'w'),indent=2,ensure_ascii=False)
print(f"\n  wrote {OUT}: {len(cur)}/{len(names)} moments")
