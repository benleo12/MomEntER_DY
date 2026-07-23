"""Compute prior moment m (signal) for all candidates, emit candidate JSONs for
tau cuts and the parameter-free S/N>1 cut. Names in rt-side×dphi-side format."""
import os,json,numpy as np,optimizer_DY_unc as o
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"4")
ENE=os.environ.get('ENERGY','13TeV'); _M={'13TeV':'13TeV','13p6TeV':'13p6TeV','13TeV_v2':'13TeV_v2','13p6TeV_v2':'13p6TeV_v2','13TeV_m':'13TeV_m','13TeV_py':'13TeV_py','13TeV_pwg':'13TeV_pwg','13TeV_lomlm':'13TeV_lomlm'}[ENE]
MOM=f"moments_{_M}"; PRIOR=f"sherpa_prior_{_M}"; CSV=f"{MOM}/DYMoments_N4LLp+N3LO.csv"
print(f"  ENERGY={ENE} prior={PRIOR} MOM={MOM}")
NEV=int(os.environ.get('NEV','20000000')); B_=2000000
mom=o.load_moments(CSV); mbp={(a,b):v for a,b,v,u in mom}; mbu={(a,b):u for a,b,v,u in mom}
ssc=o.compute_sigma_theory(CSV)
ZERO={'dphi^0','lndphi^0','rt^0','lnrt^0'}
cand=[]; seen=set()
for (a,b) in mbp:
    if a in ZERO and b in ZERO: continue
    k=tuple(sorted([a,b]));
    if k in seen: continue
    seen.add(k); cand.append((a,b))
def fam(s): f,k=s.split('^'); return f,int(k)
p=o.load_prior(PRIOR); Nf=len(p['rT']); NEV=min(NEV,Nf)
idx=np.sort(np.random.default_rng(42).choice(Nf,NEV,replace=False))
rt=p['rT'][idx].astype(float); d=p['d'][idx].astype(float); w=p['w'][idx].astype(float); del p
K=len(cand); Sw=0.0; Swf=np.zeros(K)
def fval(s,rt,lrt,d,ld):
    f,k=fam(s)
    if k==0: return np.ones(len(rt))
    return {'rt':rt**k,'lnrt':lrt**k,'dphi':d**k,'lndphi':ld**k}[f]
for a0 in range(0,NEV,B_):
    b0=min(a0+B_,NEV); rb=rt[a0:b0]; db=d[a0:b0]; wb=w[a0:b0]
    lrt=np.log(np.maximum(rb,1e-12)); ld=np.log(np.maximum(db,1e-12)); Sw+=wb.sum()
    for k,(A,B) in enumerate(cand):
        Swf[k]+=wb@(fval(A,rb,lrt,db,ld)*fval(B,rb,lrt,db,ld))
m=Swf/Sw
def to_name(A,B):
    rtp=[];dp=[]
    for s in (A,B):
        f,k=fam(s)
        if k==0: continue
        (rtp if f in('rt','lnrt') else dp).append(s)
    return f"{'*'.join(rtp) if rtp else 'const^0'}×{'*'.join(dp) if dp else 'const^0'}"
def emit(keep,tag):
    names=[]; sn=set()
    for i,(A,B) in enumerate(cand):
        if not keep[i]: continue
        nm=to_name(A,B)
        if nm=='const^0×const^0' or nm in sn: continue
        sn.add(nm); names.append(nm)
    json.dump({'selected_moments':names,'n_selected':len(names),'source':tag},open(f"{MOM}/cand_{tag}.json",'w'),indent=2)
    print(f"  {tag}: {len(names)} moments")
sigth=np.array([ (ssc.get((A,B),ssc.get((B,A),0.0))**2 + mbu.get((A,B),mbu.get((B,A),0.0))**2)**0.5 for (A,B) in cand])
T=np.array([mbp.get((A,B),mbp.get((B,A),np.nan)) for (A,B) in cand])
sigth_rel=sigth/np.abs(T); signal_rel=np.abs(T-m)/np.abs(T)
emit(sigth_rel<0.010,'tau1.0')
emit(sigth_rel<0.020,'tau2.0')
emit(signal_rel>sigth_rel,'SN')   # parameter-free: signal > theory noise
print(f"  (S/N>1 = keep where |T-prior| > sigma_theory)")
