"""Grow-mode selection under a conditioning constraint (prior-universal fallback branch).

    python grow_select.py <prior_dir> <out.json> <tol> <cand1.json> [cand2.json ...]

Candidates = ordered, deduplicated union of the given pools' selected_moments (first file
first, so the model-selection winner's ordering leads). A candidate is admitted iff its
standardized feature column keeps relative residual > tol against the span of those already
admitted -- the same test rank_prune.py applies, but grown over the FULL pool instead of
pruned within one set.
"""
import sys, json, numpy as np, optimizer_DY_unc as o
from apply_lambdas import features
PRI, OUT, TOL = sys.argv[1], sys.argv[2], float(sys.argv[3])
cands=[]
for f in sys.argv[4:]:
    for n in (json.load(open(f)).get('selected_moments') or []):
        if n not in cands: cands.append(n)
p=o.load_prior(PRI); N=min(len(p['w']),1000000)
rt=(p['pT'][:N]/p['m'][:N]).astype(float); dd=p['d'][:N].astype(float)
w=np.abs(p['w'][:N].astype(float)); del p
PHI=features(cands, rt, dd)
X=(PHI-np.average(PHI,axis=0,weights=w))*np.sqrt(w/w.sum())[:,None]
X=X/np.maximum(np.linalg.norm(X,axis=0),1e-300)
kept,Q=[],[]
for k,nm in enumerate(cands):
    v=X[:,k].copy()
    for q in Q: v-=(q@v)*q
    r=np.linalg.norm(v)
    if r>TOL: kept.append(nm); Q.append(v/r)
print(f"grow: {len(kept)}/{len(cands)} candidates admitted (tol={TOL:g})")
json.dump({'selected_moments':kept,
           'grow_select':{'tol':TOL,'pool':len(cands),'sources':sys.argv[4:]}},
          open(OUT,'w'), indent=1)
print(f"wrote {OUT}")
