"""Minimal example: reweight your own Drell-Yan events to N4LL'+N3LO accuracy.

You provide four per-event arrays; you get back one reweighted weight per event.
Nothing else is needed -- the reweighting is event-local and reads only the
delivered products/<energy>/lambda_export.json.

    python examples/apply_quickstart.py      # from the repository root
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from apply_lambdas import reweight, reweight_scheme, schemes

# --- your events (here: a toy sample; replace with your generator output) -------
rng = np.random.default_rng(1)
N     = 500_000
w0    = np.ones(N)                       # generator weights
m_ll  = rng.uniform(66, 116, N)          # dilepton invariant mass [GeV]
qT    = rng.exponential(15, N)           # dilepton pT [GeV]
dphi  = rng.uniform(1e-3, 3.0, N)        # acoplanarity  d = pi - Delta phi_ll

# --- central reweighting --------------------------------------------------------
w = reweight(w0, qT, m_ll, dphi, energy="13TeV")
print(f"central: sum(w)/sum(w0) = {w.sum()/w0.sum():.4f}  (=1 on the real prior)")

# --- the theory uncertainty band: one weight set per scale/NP scheme ------------
names = schemes("13TeV")                 # ['central','2MuR','0p5MuR', ... 29 total]
band  = np.stack([reweight_scheme(w0, qT, m_ll, dphi, scheme=s, energy="13TeV")
                  for s in names])        # shape (29, N)
print(f"{len(names)} schemes -> per-event envelope is the theory band")

# --- a reweighted observable, e.g. <qT> below the hand-off ----------------------
lo = qT < 120
print(f"<qT>  prior     = {np.average(qT[lo], weights=w0[lo]):.3f} GeV")
print(f"<qT>  reweighted= {np.average(qT[lo], weights=w[lo]):.3f} GeV")
