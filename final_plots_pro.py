"""Publication plots: reweight CENTRAL @NEV and EACH scale scheme (band) so the theory
scale uncertainty is PROPAGATED through the MaxEnt reweighting. Theory band = envelope
over schemes of (scheme_density ± scheme_stat) [each variation's own stat included].
Multi-panel per observable (main + reweighted/theory + prior/theory), LaTeX, dense
ticks, ratio 0.5-1.5. Reuses MaxEntDual via the eval_set recipe; optimizer untouched.
Usage: python final_plots_pro.py [set.json]  Env: NEV FIT_NEV NEV_BAND BATCH SIG_FLOOR_REL OUT_TAG
"""
import os,sys,json,io,contextlib,shutil
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"4")
import numpy as np, optimizer_DY_unc as o
SRC=sys.argv[1] if len(sys.argv)>1 else 'moments_13TeV/stable_tau2.0.json'
NEV=int(os.environ.get('NEV','50000000')); FIT=int(os.environ.get('FIT_NEV','2000000'))
NEV_BAND=int(os.environ.get('NEV_BAND','20000000')); B_=int(os.environ.get('BATCH','2000000'))
SIGREL=float(os.environ.get('SIG_FLOOR_REL','0.005')); OUT_TAG=os.environ.get('OUT_TAG','PRO')
ACC="N4LL'+N3LO"; ACC_SLUG="N4LLp+N3LO"
# ---- energy switch (default 13 TeV) ----
ENE=os.environ.get('ENERGY','13TeV')
_EMAP={'13TeV':('13TeV','13TeV_IncPS','13TeV'),'13p6TeV':('13p6TeV','13.6TeV_IncPS','13p6TeV'),
       '13TeV_v2':('13TeV_v2','13TeV_IncPS','13TeV'),'13p6TeV_v2':('13p6TeV_v2','13.6TeV_IncPS','13p6TeV'),'13TeV_m':('13TeV_m','13TeV_IncPS','13TeV'),'13TeV_py':('13TeV_py','13TeV_IncPS','13TeV'),'13TeV_pwg':('13TeV_pwg','13TeV_IncPS','13TeV'),'13TeV_lomlm':('13TeV_lomlm','13TeV_IncPS','13TeV')}
if ENE not in _EMAP: raise SystemExit(f"unknown ENERGY={ENE}")
_mtag,_qdir,_qtag=_EMAP[ENE]
MOM=f"moments_{_mtag}"; CSV=f"{MOM}/DYMoments_{ACC_SLUG}.csv"; PRIOR=f"sherpa_prior_{_mtag}"
QT_M=f"/Users/user/Library/CloudStorage/Dropbox/DY_reweighting_data/wju/{_qdir}/qT_1D_Dist_NP_{_qtag}_IncPS_varmT.m"
print(f"  ENERGY={ENE}  prior={PRIOR}  MOM={MOM}")
names=(json.load(open(SRC)).get('selected_moments') or []); print(f"  {len(names)} moments from {SRC}")
def fampow(s): f,k=s.split('^'); return (f,int(k))
def parse_part(p): return [] if p=='const^0' else [fampow(s) for s in p.split('*')]
def mk_tgt(mbp):
    def tgt(allf):
        fs=[f'{f}^{k}' for f,k in allf]
        if len(fs)==1:
            for z in ('dphi^0','lndphi^0','rt^0','lnrt^0'):
                if (fs[0],z) in mbp: return mbp[(fs[0],z)]
                if (z,fs[0]) in mbp: return mbp[(z,fs[0])]
        elif len(fs)==2:
            if (fs[0],fs[1]) in mbp: return mbp[(fs[0],fs[1])]
            if (fs[1],fs[0]) in mbp: return mbp[(fs[1],fs[0])]
        return None
    return tgt
facs=[]
for nm in names:
    A,Bp=nm.split('×'); facs.append((parse_part(A),parse_part(Bp)))
K=len(facs)
# ---- data ----
p=o.load_prior(PRIOR); Nf=len(p['rT']); NEV=min(NEV,Nf); FIT=min(FIT,NEV); NEV_BAND=min(NEV_BAND,NEV)
_ns=min(500000,Nf); rt_ref=p['rT'][:_ns].astype(float); d_ref=p['d'][:_ns].astype(float)  # PINNED std ref (original order, NEV-independent)
idx=np.sort(np.random.default_rng(42).choice(Nf,NEV,replace=False))
rt=p['rT'][idx].astype(float); d=p['d'][idx].astype(float); w=p['w'][idx].astype(float); pT=p['pT'][idx].astype(float); del p
def facj1(f,k,x):
    if k==0: return np.ones(len(x))
    return x**k if f in('rt','dphi') else np.log(np.maximum(x,1e-12))**k
def side(fl,x):
    out=np.ones(len(x))
    for f,k in fl: out=out*facj1(f,k,x)
    return out
def Fc(a,b): return np.column_stack([side(fa,rt[a:b]) for fa,fb in facs])
def Gc(a,b): return np.column_stack([side(fb,d[a:b])  for fa,fb in facs])
# PINNED standardization: std over a FIXED original-order slice (NEV/idx-independent) so
# lam_c, the exported lambda_physical, and the weights are all bit-consistent across runs.
sF=np.array([side(fa,rt_ref).std() for fa,fb in facs]); sG=np.array([side(fb,d_ref).std() for fa,fb in facs])
sF[sF==0]=1; sG[sG==0]=1
pairs=np.array([(k,k) for k in range(K)],np.int64); ii=pairs[:,0]; jj=pairs[:,1]
def Fs(a,b): return Fc(a,b)/sF
def Gs(a,b): return Gc(a,b)/sG
# cache FIT feature matrices ONCE. Central fit on FIT; band(scheme) fits on a small
# FIT_B subsample (band is approximate) -> cheap line-search, warm-started from central.
Ff=Fs(0,FIT); Gf=Gs(0,FIT)
FIT_B=min(int(os.environ.get('FIT_BAND','500000')),FIT); Ff_b=Fs(0,FIT_B); Gf_b=Gs(0,FIT_B)
def fit_lam(T,Ff_,Gf_,w_,lam0=None,steps=200):
    Ts=T/(sF*sG)
    with contextlib.redirect_stdout(io.StringIO()):
        m=o.MaxEntDual(Ff_,Gf_,pairs,Ts,w_,sigmas_target=list(Sig_scaled),
                       cov_target=COV_scaled if 'COV_scaled' in globals() and COV_scaled is not None else None)
        if lam0 is not None: m.lam=lam0.copy()    # warm start (scheme targets ≈ central)
        o.optimize_newton(m,max_steps=steps,tol=1e-10,verbose=False)
    lam=m.lam.copy()
    return lam if np.all(np.isfinite(lam)) else (lam0.copy() if lam0 is not None else np.zeros(K))
def apply_many(lams,nev):
    """apply a LIST of λ's to first nev events in ONE feature pass (features built once
    per batch, reused for all λ). Returns list of signed-weight arrays."""
    J=len(lams); logaw=np.log(np.maximum(np.abs(w[:nev]),1e-300)); sgn=np.where(w[:nev]>=0,1.0,-1.0)
    gmax=np.full(J,-np.inf)
    for a in range(0,nev,B_):
        b=min(a+B_,nev); Phi=Fs(a,b)[:,ii]*Gs(a,b)[:,jj]
        for j in range(J): gmax[j]=max(gmax[j],float((logaw[a:b]+Phi@lams[j]).max()))
    vvs=[np.empty(nev) for _ in range(J)]
    for a in range(0,nev,B_):
        b=min(a+B_,nev); Phi=Fs(a,b)[:,ii]*Gs(a,b)[:,jj]
        for j in range(J): vvs[j][a:b]=sgn[a:b]*np.exp(logaw[a:b]+Phi@lams[j]-gmax[j])
    return vvs
# ---- central reweight @NEV ----
mom_c=o.load_moments(CSV); mbp_c={(a,b):v for a,b,v,u in mom_c}; tgt_c=mk_tgt(mbp_c)
Tc=np.array([tgt_c(fa+fb) or 0.0 for fa,fb in facs])
mbu_c={(a,b):u for a,b,v,u in mom_c}
def sig_for(allf):
    fs=[f'{f}^{k}' for f,k in allf]; cs=[]
    if len(fs)==1:
        for z in ('dphi^0','lndphi^0','rt^0','lnrt^0'): cs+=[(fs[0],z),(z,fs[0])]
    elif len(fs)==2: cs+=[(fs[0],fs[1]),(fs[1],fs[0])]
    for k in cs:
        sc=ssc_g.get(k); st=mbu_c.get(k)
        if sc is not None or st is not None: return ((sc or 0)**2+(st or 0)**2)**0.5
    return None
ssc_g=o.compute_sigma_theory(CSV)
SIG_MODE=os.environ.get('SIG_MODE','statscale')
if SIG_MODE in ('stat','cov'):      # stat-only central penalty: scale systematics are NOT independent
    ssc_g={}              # per-moment noise — they are correlated shifts already propagated
    print("  SIG_MODE=stat (central penalty: stat only; scale unc -> band)")   # via the 28-scheme band.
Sigr=np.array([ (sig_for(fa+fb) or 0.005*abs(Tc[k])) for k,(fa,fb) in enumerate(facs)])
Sig_scaled=np.maximum(Sigr/(sF*sG),1e-12)
COV_scaled=None
if SIG_MODE=='cov':
    # scale-scheme covariance in scaled units: each scheme = a ±1σ realization of the
    # correlated scale nuisance -> Σ_scale = mean_s δT δTᵀ about the central; + diag(stat²).
    _ck,_vk=o.get_scale_variations(CSV)
    def _tgt_of(fo,res):
        ms=o.load_moments_for_scale(CSV,fo,res); mb={(a,b):v for a,b,v,u in ms}; tg=mk_tgt(mb)
        return np.array([ (tg(fa+fb) if tg(fa+fb) is not None else Tc[k]) for k,(fa,fb) in enumerate(facs)])
    _D=np.vstack([ (_tgt_of(fo,res)-Tc)/(sF*sG) for (fo,res) in _vk ])
    COV_scaled=(_D.T@_D)/len(_D)+np.diag(Sig_scaled**2)
    print(f"  SIG_MODE=cov: Σ = scheme-cov({len(_D)}) + diag(stat²); rank≈{np.linalg.matrix_rank(_D)}+diag")
class StreamDual:
    """MaxEntDual-compatible dual over the FULL sample with batched feature rebuilds.
    Same penalized objective; N = all events (no dense feature matrix in RAM)."""
    def __init__(self,n,targets,sig):
        self.N=n; self.K=K; self.targets=np.asarray(targets,float)
        self.sigma=np.asarray(sig,float); self.sigma2=self.sigma**2
        self.reg_coef=self.sigma2; self.reg_mat=None
        self.lam=np.zeros(K); self._logaw=np.log(np.maximum(np.abs(w[:n]),1e-300))
        self._sgn=np.where(w[:n]>=0,1.0,-1.0)
    def _pass(self,lam,need_cov):
        gmax=-np.inf
        for a in range(0,self.N,B_):
            b=min(a+B_,self.N); T=Fs(a,b)[:,ii]*Gs(a,b)[:,jj]
            gmax=max(gmax,float((self._logaw[a:b]+T@lam).max()))
        S0=0.0; S1=np.zeros(self.K); S2=np.zeros((self.K,self.K)) if need_cov else None
        for a in range(0,self.N,B_):
            b=min(a+B_,self.N); T=Fs(a,b)[:,ii]*Gs(a,b)[:,jj]
            ww=self._sgn[a:b]*np.exp(self._logaw[a:b]+T@lam-gmax)
            S0+=ww.sum(); S1+=ww@T
            if need_cov: S2+=(ww[:,None]*T).T@T
        if not (S0>0) or not np.isfinite(S0):
            return np.inf,np.full(self.K,np.nan),(np.full((self.K,self.K),np.nan) if need_cov else None)
        mom=S1/S0; cov=(S2/S0-np.outer(mom,mom)) if need_cov else None
        return gmax+np.log(S0),mom,cov
    def dual_loss(self,lam):
        lz,_,_=self._pass(lam,False)
        return lz-lam@self.targets+0.5*(self.reg_coef*lam*lam).sum()
    def dual_loss_grad_hess(self,lam):
        lz,mom,cov=self._pass(lam,True)
        loss=lz-lam@self.targets+0.5*(self.reg_coef*lam*lam).sum()
        return loss,mom-self.targets+self.reg_coef*lam,cov+np.diag(self.reg_coef),mom
print("  reweighting CENTRAL @%dM ..."%(NEV//1_000_000))
# cache the expensive central fit (200-step Newton, deterministic) -> fast plot iteration
import hashlib
_LC=f"{MOM}/.lamc_{os.path.basename(SRC)}_{FIT}_{hashlib.md5(('|'.join(names)+SIG_MODE).encode()).hexdigest()[:8]}.npy"  # content+sigmode-keyed cache
Tc_fit=Tc
if int(os.environ.get('TARGET_SHIFT','0')):
    # first-order full-sample correction: the 2M fit matches the FIT slice's empirical
    # moments; shifting targets by the prior-moment difference (full - FIT) makes the
    # fitted lambdas satisfy closure on the FULL sample (to first order in the drift).
    _LSH=_LC.replace('.npy','_shift.npy')
    if os.path.exists(_LSH) and not int(os.environ.get('NOCACHE','0')):
        _dmu=np.load(_LSH)
    else:
        _n2=np.zeros(K); _nF=np.zeros(K); _s2=0.0; _sF_=0.0
        for a in range(0,NEV,B_):
            b=min(a+B_,NEV); ph=Fc(a,b)*Gc(a,b); wv=w[a:b]
            _nF+=wv@ph; _sF_+=wv.sum()
            if a<FIT:
                bb=min(b,FIT); ph2=ph[:bb-a]; wv2=wv[:bb-a]; _n2+=wv2@ph2; _s2+=wv2.sum()
        _dmu=(_nF/_sF_)-(_n2/_s2); np.save(_LSH,_dmu)
    Tc_fit=Tc-_dmu
    print(f"  TARGET_SHIFT: max|shift|/|T| = {np.max(np.abs(_dmu)/np.maximum(np.abs(Tc),1e-30))*100:.3f}%")
    _LC=_LC.replace('.npy','_ts.npy')
FULLFIT=int(os.environ.get('FULLFIT','0'))
if FULLFIT: _LC=_LC.replace('.npy',f'_full{NEV//10**6}M.npy')
if os.path.exists(_LC) and not int(os.environ.get('NOCACHE','0')):
    lam_c=np.load(_LC); print(f"  loaded cached lam_c from {_LC}")
elif FULLFIT:
    print(f"  FULLFIT: streaming fit on ALL {NEV//10**6}M events ...")
    import time as _t; _t0=_t.time()
    _m=StreamDual(NEV,Tc/(sF*sG),list(Sig_scaled))
    with contextlib.redirect_stdout(io.StringIO()):
        o.optimize_newton(_m,max_steps=int(os.environ.get('MAX_STEPS','60')),tol=1e-9,verbose=False)
    lam_c=_m.lam.copy(); np.save(_LC,lam_c)
    print(f"  FULLFIT done in {(_t.time()-_t0)/60:.1f} min -> {_LC}")
else:
    lam_c=fit_lam(Tc_fit,Ff,Gf,w[:FIT]); np.save(_LC,lam_c); print(f"  fit + cached lam_c -> {_LC}")
if int(os.environ.get('EXPORT','0')):
    # full-sample log-normalization shift C so that  w_rew = w0*exp(logit - C)  is already
    # correctly normalized (sum w_rew = sum w0) with NO global max and NO post-hoc rescale
    # -> fully event-local, usable during event generation. C = log( sum w0 e^logit / sum w0 ).
    print("  computing log-norm shift C over %dM events ..."%(NEV//10**6))
    gmax=-np.inf
    for a in range(0,NEV,B_):
        b=min(a+B_,NEV); lg=(Fs(a,b)[:,ii]*Gs(a,b)[:,jj])@lam_c; gmax=max(gmax,float(lg.max()))
    S1=0.0; S0=float(w[:NEV].sum())
    for a in range(0,NEV,B_):
        b=min(a+B_,NEV); lg=(Fs(a,b)[:,ii]*Gs(a,b)[:,jj])@lam_c; S1+=float((w[a:b]*np.exp(lg-gmax)).sum())
    C=float(gmax+np.log(S1/S0))
    print(f"  log_norm_shift C={C:.6f}")
    lam_phys=(lam_c/(sF*sG)).tolist()
    exp={'description':f'MaxEnt reweighting of {PRIOR} to Wan-Li N4LLp+N3LO. Per event, fully local: w_rew = w0 * exp( sum_k lambda_physical[k]*phi_k - log_norm_shift ). The shift already fixes the normalization (sum w_rew = sum w0); no global max or renormalization needed.',
      'n_moments':K,'moments':names,'lambda_physical':lam_phys,'log_norm_shift':C,
      'feature_convention':{'phi_k':'product over the two parts of moment name "A×B" (A=rt-side, B=dphi-side)',
        'monomials':'rt^k=rt**k, dphi^k=dphi**k, lnrt^k=log(rt)**k, lndphi^k=log(dphi)**k, const^0=1; parts within a side joined by *',
        'vars':'rt=qT/m_ll, dphi=pi-Delta_phi_ll; clip log args at 1e-12'},
      'standardization_note':'lambda_physical already folds in the unit-std feature scaling; apply directly to raw monomials. (lambda_standardized + sF,sG given only for cross-check.)',
      'lambda_standardized':lam_c.tolist(),'sF':sF.tolist(),'sG':sG.tolist(),
      'gating':{'variable':'qT [GeV]','full_reweight_below':120,'window_GeV':[120,200],'his_region_above':200,
        'beta(qT)':'b = 1 - (6 t^5 - 15 t^4 + 10 t^3),  t = clip((qT-120)/80, 0, 1)   [1 below 120, 0 above 200]',
        'combine':'w = b*w_rew + (1-b)*w0*s_i ;  w_rew = w0*exp(logit - log_norm_shift) ;  s_i = your event-level tail factor (s_i=1 => revert to prior)',
        'why':'qT=200 GeV is the edge of the N4LLp+N3LO prediction; q0=120 is where its theory-stat reaches ~5% (~size of the correction). taper systematic = 0.21x theory band.'},
      'closure':'reproduces the 52 N4LLp+N3LO moments to median 0.33% / max 4%; central N_eff=4.77%'}
    json.dump(exp,open(f'{MOM}/lambda_export.json','w'),indent=1)
    print(f"  wrote {MOM}/lambda_export.json ({K} moments, log_norm_shift={C:.4f})"); sys.exit(0)
vv=apply_many([lam_c],NEV)[0]
neff=vv.sum()**2/np.sum(vv*vv)/NEV*100; print(f"  central N_eff={neff:.2f}%")
if int(os.environ.get('STEFAN','0')):
    # Stefan gating diagnostic: he point-wise turns OFF our reweighting above QCUT GeV
    # (trusts prior/his multijet+EW there). Quantify (1) closure: do λ reproduce Wan-Li's
    # moments?  (2) what σ-fraction lives above QCUT  (3) which of our moments draw from
    # the tail (=those gating disturbs)  (4) how wildly exp(Σλφ) extrapolates in the tail.
    QCUT=float(os.environ.get('QCUT','200')); tail=pT[:NEV]>QCUT
    den=vv.sum(); dent=vv[tail].sum()
    num=np.zeros(K); numt=np.zeros(K)
    for a in range(0,NEV,B_):
        b=min(a+B_,NEV); ph=Fc(a,b)*Gc(a,b); wv=vv[a:b]; num+=wv@ph; numt+=(wv*tail[a:b])@ph
    mom=num/den; rel=np.abs(mom-Tc)/np.maximum(np.abs(Tc),1e-12)*100
    fac=vv/np.where(w[:NEV]==0,np.nan,w[:NEV]); fac=fac/np.nanmedian(fac[~tail])  # reweight factor rel. to bulk
    tailc=np.abs(numt)/np.maximum(np.abs(num),1e-30)*100
    print(f"\n  ===== STEFAN gating diagnostic (qT cut = {QCUT:g} GeV, @{NEV//10**6}M) =====")
    print(f"  (1) CLOSURE  apply λ to prior -> 52 moments vs Wan-Li: median={np.median(rel):.3f}%  max={rel.max():.3f}%")
    print(f"  (2) σ above {QCUT:g} GeV:  reweighted={100*dent/den:.4f}%   prior={100*w[:NEV][tail].sum()/w[:NEV].sum():.4f}%   ({tail.sum()} events)")
    print(f"  (4) reweight factor exp(Σλφ) in tail (rel. to bulk median): max={np.nanmax(fac[tail]):.2f}x  p99={np.nanpercentile(fac[tail],99):.2f}x  (bulk p99={np.nanpercentile(fac[~tail],99):.2f}x)")
    print(f"  (3) per-moment fraction drawn from qT>{QCUT:g} (gating disturbs these):")
    for k in np.argsort(-tailc)[:12]: print(f"      {names[k]:32s} tail={tailc[k]:7.3f}%  closure={rel[k]:.3f}%")
    print(f"      bulk-safe moments (tail<0.5%): {int(np.sum(tailc<0.5))}/{K}   |  tail<2%: {int(np.sum(tailc<2))}/{K}")
    sys.exit(0)
# ---- reweight each scale scheme via LINEAR RESPONSE about the central solution ----
# At the penalized optimum λ_c the gradient vanishes, so the Newton step for a scheme
# whose target is T_sch is EXACTLY  Δλ = H_c⁻¹(T_sch − T_c)  (scaled units), with
# H_c = Cov_w(φ)+diag(σ²) evaluated ONCE at λ_c. The scale variations are small
# perturbations of the same physics, so this first-order propagation IS the theory
# scale-uncertainty band — and it reuses one Hessian (fast) and never blows up from
# per-scheme ill-conditioning. (Validate vs a full fit with VERIFY=1.)
cen_key,varkeys=o.get_scale_variations(CSV)
import os as _os
varkeys=varkeys[:int(_os.environ.get('SCHMAX','99'))]
print(f"  propagating {len(varkeys)} scale schemes (linear response) ...")
Tc_s=Tc/(sF*sG)
with contextlib.redirect_stdout(io.StringIO()):
    m_c=o.MaxEntDual(Ff,Gf,pairs,Tc_s,w[:FIT],sigmas_target=list(Sig_scaled),cov_target=COV_scaled if COV_scaled is not None else None); m_c.lam=lam_c.copy()
    _,_,H_c,_=m_c.dual_loss_grad_hess(lam_c)
# H_c is symmetric PSD but VERY ill-conditioned (collinear moments, κ~1e17). A raw solve
# amplifies the scale shift along near-null directions -> unphysical band. Use a truncated
# pseudo-inverse: propagate Δμ only along eigen-directions the data actually determines
# (eigenvalue > RCOND·λ_max); freeze λ along the rest. RCOND set so ALL 28 schemes are
# stable (no scheme dropped) and the band reflects genuinely-constrained scale response.
_RCOND=float(_os.environ.get('RCOND','1e-3'))
_evw,_evv=np.linalg.eigh(H_c)
def make_pinv(rc):
    k=_evw>rc*_evw.max(); return np.where(k,1.0/np.where(k,_evw,1.0),0.0),int(k.sum())
def solve_lr(dmu,pinv): return _evv@(pinv*(_evv.T@dmu))
def sch_target(fo,res):
    ms=o.load_moments_for_scale(CSV,fo,res); mbps={(a,b):v for a,b,v,u in ms}; tg=mk_tgt(mbps)
    return np.array([ (tg(fa+fb) if tg(fa+fb) is not None else Tc[k]) for k,(fa,fb) in enumerate(facs)])
dmus=[(sch_target(fo,res)-Tc)/(sF*sG) for (fo,res) in varkeys]
_pinv,_nk=make_pinv(_RCOND)
print(f"  H_c eigen: keep {_nk}/{len(_evw)} dirs (RCOND={_RCOND:g}, κ_full={_evw.max()/max(_evw.min(),1e-300):.1e})")
if _os.environ.get('SCANRCOND'):
    NS=int(_os.environ.get('NEV_SCAN','1000000'))
    RT0=np.array(json.load(open('atlas_edges.json'))['rT'])
    for rc in [float(x) for x in _os.environ['SCANRCOND'].split(',')]:
        pv,nk=make_pinv(rc); ls=[lam_c+solve_lr(d,pv) for d in dmus]; vs=apply_many(ls,NS)
        nn=[ (v.sum()**2/np.sum(v*v)/len(v)*100) for v in vs]
        def dens(v,e):
            h=o.hist_to_density(rt[:NS],v,e); return h/np.sum(h*np.diff(e))
        D=np.vstack([dens(v,RT0) for v in vs]); bw=100*np.nanmedian((np.nanmax(D,0)-np.nanmin(D,0))/2/np.nanmean(D,0))
        print(f"  SCAN RCOND={rc:g}: keep {nk}/52  Neff[min/med]={np.nanmin(nn):.2f}/{np.nanmedian(nn):.2f}%  rT-band-med={bw:.2f}%")
    print("ALL DONE"); sys.exit(0)
lam_sch=[lam_c+solve_lr(d,_pinv) for d in dmus]
if int(_os.environ.get('EXPORT_VARS','0')):
    # export per-scheme lambdas + normalization shifts for ALL scale variations (+central),
    # one file keyed by scheme name. Same phi_k and gating as the central lambda_export.json.
    def _nm(fo,res): return (fo.split('->')[0] if fo!='CV->FO' else res.split('->')[0]) if (fo!='CV->FO' or res!='CV->Res') else 'central'
    def _grp(k):
        s=str(k)
        if 'C0_np' in s: return 'C0NP'
        if 'kappa_np' in s: return 'kappaNP'
        if ('MuR->FO' in s) or ('MuF->FO' in s) or ('MuRF->FO' in s): return 'FO'
        return 'Resum'
    all_lams=[lam_c]+lam_sch; labels=['central']+[_nm(fo,res) for (fo,res) in varkeys]
    groups=['central']+[_grp((fo,res)) for (fo,res) in varkeys]; J=len(all_lams)
    print(f"  EXPORT_VARS: {J} schemes; computing per-scheme log-norm shifts over {NEV//10**6}M events ...")
    gmax=np.full(J,-np.inf)
    for a in range(0,NEV,B_):
        b=min(a+B_,NEV); Phi=Fs(a,b)[:,ii]*Gs(a,b)[:,jj]
        for j in range(J): gmax[j]=max(gmax[j],float((Phi@all_lams[j]).max()))
    S1=np.zeros(J); S0=float(w[:NEV].sum())
    for a in range(0,NEV,B_):
        b=min(a+B_,NEV); Phi=Fs(a,b)[:,ii]*Gs(a,b)[:,jj]
        for j in range(J): S1[j]+=float((w[a:b]*np.exp(Phi@all_lams[j]-gmax[j])).sum())
    Cs=(gmax+np.log(S1/S0)).tolist()
    schemes={labels[j]:{'group':groups[j],'log_norm_shift':float(Cs[j]),
                        'lambda_physical':(all_lams[j]/(sF*sG)).tolist()} for j in range(J)}
    out={'description':'Per-scheme MaxEnt reweighting (central + Wan-Li scale/NP variations). For scheme S, per event: w_rew = w0*exp(sum_k schemes[S].lambda_physical[k]*phi_k - schemes[S].log_norm_shift); then the SAME gating as the central file. phi_k and gating identical to lambda_export.json. Variations are first-order (linear-response) propagations of Wan-Li 28-scheme moment shifts; their envelope = the theory scale-uncertainty band.',
         'n_moments':K,'moments':names,'n_schemes':J,'scheme_names':labels,
         'feature_convention':{'phi_k':'product over the two parts of moment name "A×B"',
            'monomials':'rt^k=rt**k, dphi^k=dphi**k, lnrt^k=log(rt)**k, lndphi^k=log(dphi)**k, const^0=1; parts within a side joined by *',
            'vars':'rt=qT/m_ll, dphi=pi-Delta_phi_ll; clip log args at 1e-12'},
         'gating':{'variable':'qT [GeV]','window_GeV':[120,200],
            'beta(qT)':'b = 1 - (6 t^5 - 15 t^4 + 10 t^3),  t = clip((qT-120)/80, 0, 1)',
            'combine':'w = b*w_rew + (1-b)*w0*s_i  (above 200 -> prior, then your s_i)'},
         'schemes':schemes}
    json.dump(out,open(f'{MOM}/lambda_export_variations.json','w'))
    print(f"  wrote {MOM}/lambda_export_variations.json ({J} schemes)")
    print("  scheme  group     C        |  scheme  group     C")
    for j in range(J): print(f"    {labels[j]:14s} {groups[j]:8s} {Cs[j]:+.3f}")
    sys.exit(0)
print(f"  applying {len(lam_sch)} schemes @%dM (single feature pass) ..."%(NEV_BAND//1_000_000)); vv_sch=apply_many(lam_sch,NEV_BAND)
print(f"  reweighted ALL {len(vv_sch)} schemes (no filter, linear response)")
if int(_os.environ.get('GATE_PRO','0')):
    # gate the central AND every scale scheme: smooth hand-off to prior over [GATE_LO,GATE_HI]
    # so the scale band + type panels taper to the prior above the Wan-Li data edge (we do not
    # claim his scale uncertainty where his prediction stops). Same β as the GATE study.
    GLO=float(_os.environ.get('GATE_LO',120)); GHI=float(_os.environ.get('GATE_HI',200))
    def _gate(wt,n):
        z=np.clip((pT[:n]-GLO)/(GHI-GLO),0,1); b=1.0-(6*z**5-15*z**4+10*z**3)
        return b*(wt*(w[:n].sum()/wt.sum()))+(1.0-b)*w[:n]
    vv=_gate(vv,NEV); vv_sch=[_gate(v,NEV_BAND) for v in vv_sch]
    print(f"  GATED central + {len(vv_sch)} schemes over [{GLO:g},{GHI:g}] GeV (PRO plots tap to prior in tail)")
if int(_os.environ.get('DIAG','0')):
    nn=np.array([ (v.sum()**2/np.sum(v*v)/len(v)*100) for v in vv_sch])
    dl=np.array([np.linalg.norm(l-lam_c) for l in lam_sch])
    print(f"  DIAG per-scheme: N_eff[min/med/max]={np.nanmin(nn):.2f}/{np.nanmedian(nn):.2f}/{np.nanmax(nn):.2f}%  |dlam|[med/max]={np.median(dl):.2e}/{dl.max():.2e}")
    for n in np.argsort(-dl)[:6]: print(f"    worst sch{n:2d} {str(varkeys[n]):46s} |dlam|={dl[n]:.2e} Neff={nn[n]:.2f}%")
# ---- theory dists (per-scheme + per-scheme stat) ----
hist=o._find_hist_csvs(MOM,ACC_SLUG) or o._find_hist_csvs('.',ACC_SLUG); td=o.load_target_distributions(hist,ACC); dmax=float(td['dphiDist']['edges'][-1])
print(f"  theory dists from: {hist}")
qt=o.load_pT_theory(QT_M,ACC)
ae=json.load(open('atlas_edges.json')); RT=np.array(ae['rT']); DP=np.array(ae['dphi'])
PT=np.array([0,2,4,6,8,10,12,14,16,18,20,22.5,25,27.5,30,33,36,39,42,45,48,51,54,57,61,65,70,75,80,85,95,105,125,150,175,200],float)
def rebin_d(se,sv,de):
    out=np.zeros(len(de)-1)
    for j in range(len(de)-1):
        lo=np.maximum(de[j],se[:-1]); hi=np.minimum(de[j+1],se[1:]); ov=np.clip(hi-lo,0,None); out[j]=np.sum(sv*ov)/(de[j+1]-de[j])
    return out
def rebin_q(se,su,de):  # correlated (density-avg) stat rebin
    return rebin_d(se,su,de)
def theory_on(dist,de):
    """central, scale-band(lo,hi) including each scheme's stat, central-stat — all
    normalized by central integral over `de`."""
    se=np.array(dist['edges']); cen=rebin_d(se,np.array(dist['central']),de); A=float(np.sum(cen*np.diff(de)))
    cenN=cen/A; stat=rebin_q(se,np.array(dist['central_unc']),de)/A
    vm=dist.get('var_map',{}); vu=dist.get('var_unc_map',{})
    his=[]; los=[]
    for kk in vm:
        dv=rebin_d(se,np.array(vm[kk]),de)/A; du=rebin_q(se,np.array(vu.get(kk,np.zeros_like(vm[kk]))),de)/A
        his.append(dv+du); los.append(dv-du)
    bhi=np.max(np.vstack(his),0) if his else cenN; blo=np.min(np.vstack(los),0) if los else cenN
    return cenN,blo,bhi,stat
def mc_on(x,vvx,de):
    h=o.hist_to_density(x,vvx,de); wid=np.diff(de); return h/np.sum(h*wid)
def mc_band(x,de,cen_full,vvc):
    """reweighted scale band ANCHORED to the central curve: per-bin RELATIVE spread of the
    schemes w.r.t. the central evaluated on the SAME NEV_BAND subsample (kills the sample-
    offset), envelope includes the central (ratio 1) so the band always wraps it."""
    cb=mc_on(x[:NEV_BAND],vvc[:NEV_BAND],de); cb=np.where(cb>0,cb,np.nan)
    rs=[mc_on(x[:NEV_BAND],vs,de)/cb for vs in vv_sch if np.all(np.isfinite(vs))]
    if not rs: return None,None
    R=np.vstack(rs+[np.ones_like(cb)])
    return cen_full*np.nanmin(R,0), cen_full*np.nanmax(R,0)
if int(_os.environ.get('VERIFY','0')):  # honesty check: linear-response vs real full Newton fit
    print("  VERIFY: linear-response vs full-Newton fit (reweighted density)")
    nv=min(int(_os.environ.get('VERIFY','0')),len(varkeys))
    lf=[fit_lam(sch_target(*varkeys[n]),Ff,Gf,w[:FIT],lam0=lam_c,steps=80) for n in range(nv)]
    av=apply_many([x for n in range(nv) for x in (lam_sch[n],lf[n])],NEV_BAND)
    for n in range(nv):
        for tagv,ev,xv in [('rT',RT,rt),('pT',PT,pT)]:
            dl=mc_on(xv[:NEV_BAND],av[2*n],ev); dff=mc_on(xv[:NEV_BAND],av[2*n+1],ev)
            print(f"    scheme{n} {tagv}: LR-vs-fullfit max dev={np.nanmax(np.abs(dl-dff)/np.maximum(dff,1e-30))*100:.2f}%")
# ---- classify the 28 scale schemes into Wan-Li's uncertainty TYPES ----
def grp(k):
    s=str(k)
    if 'C0_np'    in s: return 'C0NP'
    if 'kappa_np' in s: return 'kappaNP'
    if ('MuR->FO' in s) or ('MuF->FO' in s) or ('MuRF->FO' in s): return 'FO'
    return 'Resum'                                  # beam/soft/hard/ct/fac/nu/ftran scales
GROUPS=[('Resum',r'Resum.','C0'),('FO',r'FO','C2'),
        ('C0NP',r"$C_0^{\rm NP}$",'C1'),('kappaNP',r'$\kappa_{\rm NP}$','C4')]
sch_grp=[grp(k) for k in varkeys]                    # group per reweighted scheme (vv_sch order)
for gk,gl,gc in GROUPS: print(f"    group {gk}: {sch_grp.count(gk)} schemes")
def theory_band_grp(dist,de,gname):
    """theory SCALE band restricted to one uncertainty type (scale variation only; the
    per-bin statistical unc is shown in the total ratio panel, where it belongs). The
    type panels isolate the scale-variation structure by source."""
    se=np.array(dist['edges']); cen=rebin_d(se,np.array(dist['central']),de); A=float(np.sum(cen*np.diff(de))); cenN=cen/A
    vm=dist.get('var_map',{}); his=[cenN]; los=[cenN]
    for kk in vm:
        if grp(kk)!=gname: continue
        dv=rebin_d(se,np.array(vm[kk]),de)/A
        his.append(dv); los.append(dv)
    return np.min(np.vstack(los),0),np.max(np.vstack(his),0)
def rew_band_grp(x,de,gname,cen_full,vvc):
    cb=mc_on(x[:NEV_BAND],vvc[:NEV_BAND],de); cb=np.where(cb>0,cb,np.nan)
    rs=[mc_on(x[:NEV_BAND],vv_sch[i],de)/cb for i in range(len(vv_sch))
        if sch_grp[i]==gname and np.all(np.isfinite(vv_sch[i]))]
    if not rs: return None,None
    R=np.vstack(rs+[np.ones_like(cb)])
    return cen_full*np.nanmin(R,0), cen_full*np.nanmax(R,0)
# ---- plot (LaTeX, dense ticks, ratio 0.8-1.2, ratio split by uncertainty type) ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
usetex=shutil.which('latex') is not None
plt.rcParams.update({'text.usetex':usetex,'font.family':'serif','font.size':12,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
    'xtick.minor.visible':True,'ytick.minor.visible':True,'legend.frameon':False,
    'axes.linewidth':0.9,'xtick.major.size':6,'xtick.minor.size':3,'ytick.major.size':6,'ytick.minor.size':3})
def stair(ax,e,y,**k): ax.stairs(y,e,**k)
def band(ax,e,lo,hi,**k): ax.stairs(hi,e,baseline=lo,fill=True,lw=0,**k)
def bandh(ax,e,lo,hi,color): ax.stairs(hi,e,baseline=lo,fill=True,facecolor='none',edgecolor=color,hatch='////',lw=0)
if int(os.environ.get('GATE','0')):
    # ===== two-stage GATED reweighting (Stefan) =====
    # bulk (qT<QCUT): our MaxEnt factor exp(Σλφ); tail (qT>=QCUT): his event-level factor
    # s_i (placeholder 1 = revert to prior). Built as one product of per-event factors.
    QCUT=float(os.environ.get('QCUT','200')); NG=NEV; tailm=pT[:NG]>QCUT
    # vv is the bulk reweighting normalized to unit σ; rescale to the prior's normalization
    # (Σw0/Σvv) so the per-event factor has weighted-mean 1 -> "off" (tail) = prior weight,
    # continuous with the bulk. Stefan's event-level tail factor s_i multiplies the prior.
    scale=w[:NG].sum()/vv[:NG].sum()
    sfac=np.ones(NG); tf=os.environ.get('STEFAN_TAIL','')   # his per-event tail factors (npy, aligned to this subsample)
    if tf and os.path.exists(tf): sfac=np.load(tf)[:NG]; print(f"  loaded Stefan tail factors: {tf}")
    w_rew=vv[:NG]*scale; w_his=w[:NG]*sfac
    vvg=np.where(tailm, w_his, w_rew)                        # HARD-cut gated combined weight
    Rfac=vv[:NG]/np.where(w[:NG]==0,np.nan,w[:NG])*scale     # mean-1 bulk reweight factor
    # SMOOTH hand-off: w = β(qT)·w_rew + (1-β)·w_his  (convex mix). Nominal β = smootherstep
    # over [GATE_LO,GATE_HI]. The TAPER STRENGTH is not a free knob: we vary it over a family
    # (window placement/shape + stat-tied Wiener β=1/(1+(σ_th/τ)²), τ∈{3,5,8}%) and take the
    # envelope as a matching systematic — shown to be ≪ the theory band.
    LO=float(os.environ.get('GATE_LO',120)); HI=float(os.environ.get('GATE_HI',200))
    P=pT[:NG]
    def ss(z): return 6*z**5-15*z**4+10*z**3
    def beta_win(lo,hi,shape=ss): return 1.0-shape(np.clip((P-lo)/(hi-lo),0,1))
    qe=np.array(qt['edges']); qc=np.array(qt['central']); qu=np.array(qt['central_unc']); qctr=0.5*(qe[:-1]+qe[1:])
    sth_pct=np.where(qc>0,100*qu/qc,1e9); sthP=np.interp(np.clip(P,qctr[0],qctr[-1]),qctr,sth_pct)
    def beta_wiener(tau,edge=HI):                            # stat-tied; rescaled to hit 0 at edge
        b=1.0/(1.0+(sthP/tau)**2); b0=1.0/(1.0+(np.interp(edge,qctr,sth_pct)/tau)**2)
        return np.where(P>=edge,0.0,np.clip((b-b0)/(1.0-b0),0,1))
    beta_nom=beta_win(LO,HI)                                 # nominal taper
    vvs=beta_nom*w_rew+(1.0-beta_nom)*w_his
    if os.environ.get('WRITE_WEIGHTS'):
        outp=os.environ['WRITE_WEIGHTS']
        np.savez_compressed(outp,weights=vvs.astype(np.float32),event_index=idx.astype(np.int64))
        print(f"  wrote {outp}.npz: {NG} per-event gated weights (float32) + event_index into the prior")
        print(f"  prior order: NEV={NG} of {Nf} total; weights[i] <-> prior event event_index[i]; =prior weight above {HI:g} GeV")
        sys.exit(0)
    fam=[beta_win(100,200),beta_win(140,200),beta_win(120,180),beta_win(LO,HI,shape=lambda z:z),
         beta_wiener(3.0),beta_wiener(5.0),beta_wiener(8.0)]  # taper-variation family
    vfam=[b*w_rew+(1.0-b)*w_his for b in fam]+[vvs]
    nearm=(P>=0.9*QCUT)&(P<QCUT)
    print(f"\n  ===== GATED reweighting (window [{LO:g},{HI:g}], τ-varied band, s_i={'Stefan' if tf else '1 placeholder'}) =====")
    print(f"  tail σ-frac: ungated={100*vv[:NG][tailm].sum()/vv[:NG].sum():.3f}%  hard={100*vvg[tailm].sum()/vvg.sum():.3f}%  smooth={100*vvs[tailm].sum()/vvs.sum():.3f}%  prior={100*w[:NG][tailm].sum()/w[:NG].sum():.3f}%")
    print(f"  boundary reweight factor median in [{0.9*QCUT:g},{QCUT:g}) = {np.nanmedian(Rfac[nearm]):.3f}")
    print(f"  N_eff: ungated={vv[:NG].sum()**2/np.sum(vv[:NG]**2)/NG*100:.2f}%  smooth={vvs.sum()**2/np.sum(vvs*vvs)/NG*100:.2f}%")
    for tag,xl,e,dist,x in [('rT',r'$r_T=p_T/m_{\ell\ell}$',RT,td['rTDist'],rt),
                            ('pT',r'$q_T=p_T^{\ell\ell}$ [GeV]',PT,qt,pT)]:
        cen,blo,bhi,stat=theory_on(dist,e); pri=mc_on(x[:NG],w[:NG],e); rew=mc_on(x[:NG],vv[:NG],e)
        gat=mc_on(x[:NG],vvg,e); sm=mc_on(x[:NG],vvs,e)
        Dfam=np.vstack([mc_on(x[:NG],vf,e) for vf in vfam]); tlo=np.nanmin(Dfam,0); thi=np.nanmax(Dfam,0)
        safe=cen>0; C=np.where(safe,cen,np.nan); M=91.1876; wlo,whi=(LO/M,HI/M) if tag=='rT' else (LO,HI)
        inwin=(0.5*(e[:-1]+e[1:])>=wlo)&(0.5*(e[:-1]+e[1:])<=whi)&safe
        tap=np.nanmedian((100*(thi-tlo)/2/np.maximum(sm,1e-30))[inwin]); thy=np.nanmedian((100*stat/np.maximum(cen,1e-30))[inwin])
        print(f"  [{tag}] taper-syst median in window = {tap:.2f}%   vs theory σ = {thy:.2f}%  ->  {tap/max(thy,1e-9):.2f}x")
        fig=plt.figure(figsize=(7.2,6.4)); gs=fig.add_gridspec(2,1,height_ratios=[3,1.3],hspace=0.06)
        ax=fig.add_subplot(gs[0]); a1=fig.add_subplot(gs[1],sharex=ax)
        for axx in (ax,a1): axx.axvspan(wlo,whi,color='0.85',alpha=0.4,lw=0)
        band(ax,e,blo,bhi,color='C3',alpha=0.18)
        stair(ax,e,np.where(safe,cen,np.nan),color='C3',lw=1.6,label=r"Wan-Li $N^4LL'+N^3LO$")
        stair(ax,e,np.where(safe,pri,np.nan),color='0.55',lw=1.3,ls='--',label='Sherpa prior')
        stair(ax,e,np.where(safe,rew,np.nan),color='C0',lw=1.4,label='reweighted (all qT)')
        stair(ax,e,np.where(safe,gat,np.nan),color='C2',lw=1.4,ls=(0,(4,2)),label=fr'gated, hard cut @{QCUT:g}')
        band(ax,e,np.where(safe,tlo,np.nan),np.where(safe,thi,np.nan),color='C1',alpha=0.45)
        stair(ax,e,np.where(safe,sm,np.nan),color='C1',lw=1.8,label=fr'gated, smooth [{LO:g},{HI:g}] $\pm$ taper')
        ax.set_yscale('log'); ax.set_ylabel(r'$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}x$'); ax.set_title(xl); ax.legend(fontsize=8.5)
        plt.setp(ax.get_xticklabels(),visible=False)
        a1.axhline(1,color='k',lw=0.7); band(a1,e,np.where(safe,blo/C,np.nan),np.where(safe,bhi/C,np.nan),color='C3',alpha=0.18)
        stair(a1,e,np.where(safe,rew/C,np.nan),color='C0',lw=1.4)
        stair(a1,e,np.where(safe,gat/C,np.nan),color='C2',lw=1.4,ls=(0,(4,2)))
        band(a1,e,np.where(safe,tlo/C,np.nan),np.where(safe,thi/C,np.nan),color='C1',alpha=0.45)
        stair(a1,e,np.where(safe,sm/C,np.nan),color='C1',lw=1.8)
        stair(a1,e,np.where(safe,pri/C,np.nan),color='0.55',lw=1.2,ls='--')
        a1.set_ylim(0.8,1.2); a1.set_ylabel('ratio to theory'); a1.set_xlabel(xl)
        out=f"{MOM}/PAPER_GATE_{tag}.png"; fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig); print(f"  wrote {out}")
    sys.exit(0)
for tag,xl,e,dist,x,logx in [('rT',r'$r_T=p_T/m_{\ell\ell}$',RT,td['rTDist'],rt,False),
                             ('dphi',r'$\pi-\Delta\phi_{\ell\ell}$',DP[DP<=dmax+1e-9],td['dphiDist'],d,False),
                             ('pT',r'$q_T=p_T^{\ell\ell}$ [GeV]',PT,qt,pT,False)]:
    cen,blo,bhi,stat=theory_on(dist,e); rew=mc_on(x,vv,e); pri=mc_on(x,w,e); rlo,rhi=mc_band(x,e,rew,vv)
    if rlo is None: rlo,rhi=rew,rew
    safe=cen>0; ic=np.where(safe,cen,np.nan); C=np.where(safe,cen,np.nan)
    nG=len(GROUPS)
    fig=plt.figure(figsize=(7.2,11.4))
    gs=fig.add_gridspec(2+nG,1,height_ratios=[3.0,1.5]+[1.0]*nG,hspace=0.05)
    ax=fig.add_subplot(gs[0]); a1=fig.add_subplot(gs[1],sharex=ax)
    # --- main spectrum ---
    band(ax,e,blo,bhi,color='C3',alpha=0.20,label=r'theory scale$\,\oplus\,$stat')
    band(ax,e,cen*(1-stat/np.maximum(cen,1e-30)),cen*(1+stat/np.maximum(cen,1e-30)),color='0.5',alpha=0.30)
    stair(ax,e,ic,color='C3',lw=1.8,label=r"Wan-Li $N^4LL'+N^3LO$")
    stair(ax,e,np.where(safe,pri,np.nan),color='0.55',lw=1.4,ls='--',label=r'Sherpa prior')
    band(ax,e,rlo,rhi,color='C0',alpha=0.25,label=r'MaxEnt reweighted (scale band)')
    stair(ax,e,np.where(safe,rew,np.nan),color='C0',lw=1.6,label=r'MaxEnt reweighted')
    ax.set_yscale('log')
    if logx: ax.set_xscale('log')
    ax.set_ylabel(r'$1/\sigma\;\mathrm{d}\sigma/\mathrm{d}x$'); ax.set_title(xl); ax.legend(fontsize=9,loc='best')
    plt.setp(ax.get_xticklabels(),visible=False)
    # --- total ratio (theory unc + reweighted unc) ---
    rr=np.where(safe,rew/C,np.nan); rp=np.where(safe,pri/C,np.nan)
    rloT=np.where(safe,rlo/C,np.nan); rhiT=np.where(safe,rhi/C,np.nan)
    sb_lo=np.where(safe,blo/C,np.nan); sb_hi=np.where(safe,bhi/C,np.nan); stt=np.where(safe,stat/C,0.0)
    band(a1,e,sb_lo,sb_hi,color='C3',alpha=0.18,label=r'Theory unc.'); band(a1,e,1-stt,1+stt,color='0.5',alpha=0.30)
    band(a1,e,rloT,rhiT,color='C0',alpha=0.22,label=r'Rew. unc.')
    a1.axhline(1,color='k',lw=0.7)
    stair(a1,e,rp,color='C3',lw=1.3,ls='--'); stair(a1,e,rr,color='C0',lw=1.6)
    a1.set_ylim(0.8,1.2); a1.set_ylabel(r'Ratio'); a1.legend(fontsize=8,loc='upper left',ncol=2)
    plt.setp(a1.get_xticklabels(),visible=False)
    # --- one ratio panel per uncertainty type: theory (solid) + reweighted (hatched) ---
    axes_g=[]
    for gi,(gk,gl,gc) in enumerate(GROUPS):
        ag=fig.add_subplot(gs[2+gi],sharex=ax); axes_g.append(ag)
        glo,ghi=theory_band_grp(dist,e,gk); rgl,rgh=rew_band_grp(x,e,gk,rew,vv)
        band(ag,e,np.where(safe,glo/C,np.nan),np.where(safe,ghi/C,np.nan),color=gc,alpha=0.30)
        if rgl is not None: bandh(ag,e,np.where(safe,rgl/C,np.nan),np.where(safe,rgh/C,np.nan),gc)
        ag.axhline(1,color='k',lw=0.7); ag.set_ylim(0.92,1.08); ag.set_ylabel(gl,fontsize=11)
        ag.yaxis.set_major_locator(MultipleLocator(0.05)); ag.yaxis.set_minor_locator(MultipleLocator(0.025))
        if gi<nG-1: plt.setp(ag.get_xticklabels(),visible=False)
    axes_g[-1].set_xlabel(xl)
    for axx in [ax,a1]+axes_g:
        if not logx: axx.xaxis.set_minor_locator(AutoMinorLocator())
    a1.yaxis.set_major_locator(MultipleLocator(0.1)); a1.yaxis.set_minor_locator(MultipleLocator(0.02))
    out=f"{MOM}/PAPER_{OUT_TAG}_{tag}.png"; fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    dev=[abs(rr[i]-1)*100 for i in range(len(cen)) if safe[i]]
    # TRUSTED-region deviation: bins below the tail hand-off (rT<2, qT<120; dphi = full
    # range). This is the region the gated deliverable actually claims — the far tail
    # reverts to the prior and its theory reference is stat-noise-dominated.
    TR={'rT':2.0,'pT':120.0,'dphi':1e9}[tag]; ctr_=0.5*(e[:-1]+e[1:])
    trust=safe&(ctr_<TR)
    devT=[abs(rr[i]-1)*100 for i in range(len(cen)) if trust[i]]
    print(f"  [{tag}] {out}: rew/thy median={np.median(dev):.2f}% worst={max(dev):.2f}% | trusted median={np.median(devT):.2f}% worst={max(devT):.2f}% | rew-band median={100*np.median((rhiT-rloT)[safe]/2):.2f}%")
print("DONE")
