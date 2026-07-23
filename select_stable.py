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
ENE=os.environ.get('ENERGY','13TeV'); _M={'13TeV':'13TeV','13p6TeV':'13p6TeV','13TeV_v2':'13TeV_v2','13p6TeV_v2':'13p6TeV_v2','13TeV_m':'13TeV_m','13TeV_py':'13TeV_py','13TeV_pwg':'13TeV_pwg','13TeV_lomlm':'13TeV_lomlm'}[ENE]
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
ssc=o.compute_sigma_theory(CSV)
SIG_MODE=os.environ.get('SIG_MODE','statscale')
if SIG_MODE in ('stat','cov'): ssc={}; print(f"  SIG_MODE={SIG_MODE} (diag penalty = stat only)")
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
prior_neff=w.sum()**2/np.sum(w*w)
MIN_EFF=float(os.environ.get('MIN_EFF_EVENTS','0'))
print(f"  NEV={NEV:,} FIT={FIT:,}  prior N_eff={prior_neff/NEV*100:.2f}%  FRAC={FRAC}  MIN_EFF={MIN_EFF:,.0f}")
_vk=o.get_scale_variations(CSV)[1] if SIG_MODE=='cov' else None
def build(nm):
    facs=[(parse_part(n.split('×')[0]),parse_part(n.split('×')[1])) for n in nm]
    sF=np.array([side(fa,rt_ref).std() for fa,fb in facs]); sG=np.array([side(fb,d_ref).std() for fa,fb in facs]); sF[sF==0]=1; sG[sG==0]=1
    Tc=np.array([lookup(mbp,fa+fb) or 0.0 for fa,fb in facs])
    Sig=np.array([ (sig_for(fa+fb) or SIGREL*abs(Tc[k])) for k,(fa,fb) in enumerate(facs)])
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
def fit_apply(nm,lam0=None):
    facs,sF,sG,Tc,Sig,COV=build(nm); K=len(facs)
    def Fs(a,b): return np.column_stack([side(fa,rt[a:b]) for fa,fb in facs])/sF
    def Gs(a,b): return np.column_stack([side(fb,d[a:b]) for fa,fb in facs])/sG
    pairs=np.array([(k,k) for k in range(K)],np.int64); ii=pairs[:,0]; jj=pairs[:,1]
    Ts=Tc/(sF*sG); sig=list(np.maximum(Sig/(sF*sG),1e-12))
    t0=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m=o.MaxEntDual(Fs(0,FIT),Gs(0,FIT),pairs,Ts,w[:FIT],sigmas_target=sig,cov_target=COV)
        if lam0 is not None and len(lam0)==K: m.lam=lam0.copy()
        o.optimize_newton(m,max_steps=int(os.environ.get('MAX_STEPS','120')),tol=1e-9,verbose=False)
    lam=m.lam.copy(); tfit=time.time()-t0
    if not np.all(np.isfinite(lam)): return 0.0,None,None,None
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
        contrib=np.array([lam[k]*(side(facs[k][0],rt[ev:ev+1])[0]/sF[k])*(side(facs[k][1],d[ev:ev+1])[0]/sG[k]) for k in range(K)])
        order=np.argsort(-np.abs(contrib)); wl=[int(k) for k in order if abs(contrib[k])>5.0][:5] or [int(order[0])]
        return 0.0, wl, (float(logit[ev]),float(rt[ev]),float(d[ev])), lam
    C=gmax+np.log(S1u/S0)                      # normalization shift
    if float(np.max(np.abs(logit-C)))>LOGIT_CAP:   # overflow guard fails -> unstable
        neff=0.0
    else:
        wg=BETA*w*np.exp(logit-C)+(1.0-BETA)*w
        S1=float(wg.sum()); S2=float((wg*wg).sum()); neff=(S1*S1/S2) if S2>0 else 0.0
    # per-moment contribution at the worst-logit event -> which moment destabilizes most
    contrib=np.array([lam[k]*(side(facs[k][0],rt[ev:ev+1])[0]/sF[k])*(side(facs[k][1],d[ev:ev+1])[0]/sG[k]) for k in range(K)])
    order=np.argsort(-np.abs(contrib))
    # batch-drop: all moments contributing |>5| to the worst event's logit (cap 5/iter);
    # at least the single top contributor. Converges to single drops near stability.
    worst=[int(k) for k in order if abs(contrib[k])>5.0][:5] or [int(order[0])]
    print(f'      [fit {tfit:.0f}s, apply {tapp:.0f}s]',flush=True)
    return neff, worst, (float(logit[ev]),float(rt[ev]),float(d[ev])), lam
cur=list(names)
lam0=None   # iter0 cold-starts (like the run); drop-iters warm-start from previous solution
for it in range(MAXDROP+1):
    neff,worst,ev,lam=fit_apply(cur,lam0)
    frac=neff/(FRAC*prior_neff) if prior_neff>0 else 0
    _nd=(MIN_EFF if MIN_EFF>0 else FRAC*prior_neff)
    print(f"  iter {it}: {len(cur)} moments  N_eff={neff/NEV*100:.3f}% ({neff:,.0f} eff-ev, need>={_nd:,.0f})  max-logit={ev[0]:.1f} @rt={ev[1]:.2f},d={ev[2]:.2e}")
    # Stability floor. Physically what matters is the ABSOLUTE number of effective events
    # available for predictions, not a fraction of the prior's own N_eff -- the relative
    # form punishes clean (unit-weight) priors, which start at N_eff=100%, far harder than
    # signed-weight ones. MIN_EFF>0 selects the absolute criterion.
    need = MIN_EFF if MIN_EFF>0 else FRAC*prior_neff
    if neff>=need:
        print(f"  STABLE at {len(cur)} moments.  (N_eff={neff:,.0f} eff-events >= {need:,.0f})"); break
    for k in sorted(worst,reverse=True):
        print(f"    drop: {cur[k]}")
        cur.pop(k)   # cold-start every iter: warm-start is counterproductive on this collinear landscape (57s cold vs 722s warm)
json.dump({'selected_moments':cur,'n_selected':len(cur),'source':f'stable @{NEV//10**6}M subset of {SRC} (auto prune)','FRAC':FRAC,'energy':ENE},open(OUT,'w'),indent=2)
print(f"\n  wrote {OUT}: {len(cur)}/{len(names)} moments")
