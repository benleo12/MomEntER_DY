"""Apply the MaxEnt reweighting (+ smooth tail hand-off) to ANY event sample.

Fully EVENT-LOCAL: the weight of one event depends only on that event (plus the
constants in the lambda_export.json file) -- no global maximum, no post-hoc
renormalization -- so it can be applied during event generation, one event at a
time. The per-event reweighting is

    w_rew = w0 * exp( sum_k lambda_physical[k] * phi_k  -  log_norm_shift )

where log_norm_shift is a fixed constant (stored in the file) that already sets the
normalization (sum w_rew = sum w0). Self-contained: reads only a lambda_export.json.

For each event you supply four numbers:
    w0       generator weight
    qT       dilepton pT             [GeV]
    m_ll     dilepton invariant mass [GeV]
    dphi_ll  = pi - Delta_phi_ll     (acoplanarity)
Above qT = 200 GeV the weight reverts to the prior (w0); multiply that region by
your own multijet + electroweak factor if desired.

Usage:
    from apply_lambdas import reweight
    w = reweight(w0, qT, m_ll, dphi_ll, energy="13TeV")   # numpy arrays or scalars

The delivered products live in products/<energy>/:
    products/13TeV/lambda_export.json              central weights (41 moments)
    products/13TeV/lambda_export_variations.json   central + 28 scale/NP schemes
    products/13p6TeV/...                           same at 13.6 TeV (28 moments)
"""
import os, json, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
def _default(energy, variations=False):
    fn = "lambda_export_variations.json" if variations else "lambda_export.json"
    return os.path.join(_HERE, "products", energy, fn)


def _monomial(part, rt, dphi):
    """one side of a moment name, e.g. 'rt^2', 'lnrt^1', 'const^0', 'dphi^3*lndphi^5'."""
    out = np.ones_like(rt, dtype=float)
    if part == 'const^0':
        return out
    for tok in part.split('*'):                 # parts within a side are multiplied
        fam, k = tok.split('^'); k = int(k)
        if   fam == 'rt':     out *= rt ** k
        elif fam == 'lnrt':   out *= np.log(np.maximum(rt,   1e-12)) ** k
        elif fam == 'dphi':   out *= dphi ** k
        elif fam == 'lndphi': out *= np.log(np.maximum(dphi, 1e-12)) ** k
        elif fam == 'const':  pass
        else: raise ValueError(f"unknown family {fam!r} in {part!r}")
    return out


def features(names, rt, dphi):
    """phi_k for every moment name 'A×B' = product of the monomials on both sides."""
    cols = []
    for nm in names:
        A, B = nm.split('×')
        cols.append(_monomial(A, rt, dphi) * _monomial(B, rt, dphi))
    return np.column_stack(cols)


def _apply(w0, qT, m_ll, dphi_ll, names, lam, C, gat, gate=True):
    w0, qT, m_ll, dphi_ll = map(np.asarray, (w0, qT, m_ll, dphi_ll))
    logit = features(names, qT / m_ll, dphi_ll) @ np.asarray(lam)   # sum_k lambda_k phi_k
    w_rew = w0 * np.exp(logit - C)                       # event-local: fixed shift, no renorm
    if not gate:
        return w_rew
    lo, hi = gat['window_GeV']                           # smooth hand-off [120,200] GeV
    t = np.clip((qT - lo) / (hi - lo), 0, 1)
    beta = 1.0 - (6*t**5 - 15*t**4 + 10*t**3)            # 1 below lo, 0 above hi
    return beta * w_rew + (1.0 - beta) * w0              # above hi -> w0 (apply your own factor)


def reweight(w0, qT, m_ll, dphi_ll, energy="13TeV", jpath=None, gate=True):
    """Central reweighting. `energy` in {"13TeV","13p6TeV"}, or pass an explicit jpath."""
    d = json.load(open(jpath or _default(energy)))
    return _apply(w0, qT, m_ll, dphi_ll, d['moments'],
                  d['lambda_physical'], d['log_norm_shift'], d['gating'], gate)


def reweight_scheme(w0, qT, m_ll, dphi_ll, scheme='central',
                    energy="13TeV", jpath=None, gate=True):
    """One scale/NP scheme (scheme='central','2MuR',...). Repeat over all schemes for the band."""
    d = json.load(open(jpath or _default(energy, variations=True)))
    s = d['schemes'][scheme]
    return _apply(w0, qT, m_ll, dphi_ll, d['moments'],
                  s['lambda_physical'], s['log_norm_shift'], d['gating'], gate)


def schemes(energy="13TeV", jpath=None):
    """List the available scale/NP scheme names (central + 28 variations)."""
    d = json.load(open(jpath or _default(energy, variations=True)))
    return list(d['schemes'].keys())


if __name__ == '__main__':
    # self-test: parse all moment names, run on random events, report
    for energy in ("13TeV", "13p6TeV"):
        d = json.load(open(_default(energy)))
        n = 100000
        rng = np.random.default_rng(0)
        w0 = rng.normal(1, 0.1, n); m = rng.uniform(60, 120, n)
        qT = rng.exponential(20, n); dphi = rng.uniform(1e-3, 3.0, n)
        w = reweight(w0, qT, m, dphi, energy=energy)
        print(f"[{energy}] parsed {len(d['moments'])} moments OK; "
              f"finite={np.all(np.isfinite(w))}, "
              f"sum(w)/sum(w0)={w.sum()/w0.sum():.4f}, "
              f"above-200-reverts={np.allclose(w[qT>200], w0[qT>200])}, "
              f"n_schemes={len(schemes(energy))}")
