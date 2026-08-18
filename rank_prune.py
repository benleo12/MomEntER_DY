"""Prune a moment set to numerical rank on a given prior (prior-universal).

    python rank_prune.py <prior_dir> <winner.json> <out.json> [tol]

Greedy in selection order: a moment is kept iff its standardized feature column has
relative residual > tol after projection onto the span of the already-kept columns,
so the lower-order member of each degenerate family survives. tol default 1e-6
(the degeneracies found are < 1e-10; genuine near-collinear physics sits at ~1e-2).
"""
import sys, json, numpy as np, optimizer_DY_unc as o
from apply_lambdas import features
PRI, SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TOL = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6
j = json.load(open(SRC)); names = j['selected_moments']
p = o.load_prior(PRI); N = min(len(p['w']), 1000000)
rt = (p['pT'][:N]/p['m'][:N]).astype(float); dd = p['d'][:N].astype(float)
w = np.abs(p['w'][:N].astype(float)); del p
PHI = features(names, rt, dd)
mu = np.average(PHI, axis=0, weights=w)
X = (PHI-mu)*np.sqrt(w/w.sum())[:,None]
X = X/np.maximum(np.linalg.norm(X,axis=0),1e-300)      # unit columns
kept, dropped, Q = [], [], []
for k,nm in enumerate(names):
    v = X[:,k].copy()
    for q in Q: v -= (q@v)*q
    r = np.linalg.norm(v)
    if r > TOL: kept.append(nm); Q.append(v/r)
    else: dropped.append((nm, r))
print(f"rank prune: kept {len(kept)}/{len(names)}  (tol={TOL:g})")
for nm,r in dropped: print(f"  DROP {nm:45s} residual={r:.2e}")
j['selected_moments'] = kept
j['rank_prune'] = {'tol': TOL, 'dropped': [nm for nm,_ in dropped], 'n_before': len(names)}
json.dump(j, open(OUT,'w'), indent=1)
print(f"wrote {OUT}")
