"""Apply the MaxEnt reweighting (+ smooth tail hand-off) to ANY event sample.

Fully EVENT-LOCAL: the weight of one event depends only on that event (plus the
constants in the lambda_export.json file) -- no global maximum, no post-hoc
renormalization -- so it can be applied during event generation, one event at a
time. The per-event reweighting is

    w_rew = w0 * exp( sum_k lambda_physical[k] * phi_k  -  log_norm_shift )

where log_norm_shift is a fixed constant (stored in the file) that already sets the
normalization (sum w_rew = sum w0): the fit transfers the SHAPE of the calculation.
The calculation's total rate enters as the overall factor K = sigma_calc/sigma_prior
(the "rate" block of the file; per scheme K_s), applied where the calculation is in
control, i.e. to the reweighted branch below the hand-off (rate_tail=True scales the
generator's tail as well, rate=False drops K). Self-contained: reads only a lambda_export.json.

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

    from apply_lambdas import reweight_scheme, schemes, theory_band
    h = {s: histogram(reweight_scheme(w0, qT, m_ll, dphi_ll, scheme=s)) for s in schemes()}
    lo, hi = theory_band(h)          # the theory band on your histograms (per-scale quadrature)

The delivered products live in products/<energy>/:
    products/13TeV/lambda_export.json              central weights (15 moments)
    products/13TeV/lambda_export_variations.json   central + 28 scale/NP schemes
    products/13p6TeV/...                           same at 13.6 TeV (14 moments)
    products/13TeV_powheg/...                      POWHEG (ATLAS 361106 config), 19 moments, ungated
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


def _apply(w0, qT, m_ll, dphi_ll, names, lam, C, gat, gate=True, w0_tail=None, K=1.0,
           njet=None, njet_max=1, rate_tail=False):
    w0, qT, m_ll, dphi_ll = map(np.asarray, (w0, qT, m_ll, dphi_ll))
    logit = features(names, qT / m_ll, dphi_ll) @ np.asarray(lam)   # sum_k lambda_k phi_k
    w_rew = w0 * np.exp(logit - C)                       # event-local: fixed shift, no renorm
    if not gate:
        return K * w_rew
    lo, hi = gat['window_GeV']                           # smooth hand-off [120,200] GeV
    t = np.clip((qT - lo) / (hi - lo), 0, 1)
    beta = 1.0 - (6*t**5 - 15*t**4 + 10*t**3)            # 1 below lo, 0 above hi
    # Above the hand-off the sample is the generator's, so a generator variation weight
    # (e.g. one member of the 7-point muR/muF set) may be supplied for the tail branch.
    # Band recipe: the 28 theory schemes (w0_tail=None) combined per scale in quadrature by
    # theory_band() -- NOT their envelope -- plus the generator variations (scheme='central',
    # w0_tail=w0_V) combined as the generator prescribes. The theory schemes revert to the
    # central prior in the tail (resummation is off there), the generator variations act
    # only in the tail, so the two tile the phase space without double counting.
    wt = w0 if w0_tail is None else np.asarray(w0_tail)
    # The rate factor belongs where the calculation is in control.  Above the hand-off the sample is
    # the generator's, and its rate there is set by the Z+jets matrix elements, not by an inclusive
    # K-factor, so by default K multiplies the reweighted branch only (rate_tail=False).  The total
    # rate of the sample is then K x (bulk) + (tail) rather than sigma_calc.  rate_tail=True restores
    # the older behaviour of scaling every event.  Both agree when there is no hand-off (POWHEG,
    # delivered ungated) or when K is 1.
    w = (K * beta) * w_rew + (1.0 - beta) * (K * wt if rate_tail else wt)
    return _restrict(w, (K if rate_tail else 1.0) * w0, njet, njet_max)


def _restrict(w, w_prior, njet, njet_max):
    """Multiplicity bound of the region of validity.  The weight is a function of qT and the
    acoplanarity alone, so a three-jet event and a one-jet event at the same qT receive the same
    factor.  Applying it at high multiplicity would erase what distinguishes them and would replace
    the merged prediction's uncertainty, which grows with the number of jets, by one taken from a
    calculation that has at most three hard partons.  Pass njet = the multiplicity of the hard
    process (the generator's merging multiplicity, NOT the number of reconstructed jets) to give the
    weight only to events with njet <= njet_max; the rest keep the prior.  The multipliers themselves
    do not depend on the multiplicity, so the same file serves either choice.  Note that restricting
    the application changes sum(w) by the share the higher multiplicities carry."""
    if njet is None:
        return w
    return np.where(np.asarray(njet) <= njet_max, w, w_prior)


def reweight(w0, qT, m_ll, dphi_ll, energy="13TeV", jpath=None, gate=True, w0_tail=None, rate=True, njet=None, njet_max=1, rate_tail=False):
    """Central reweighting. `energy` in {"13TeV","13p6TeV"}, or pass an explicit jpath.
    `w0_tail`: optional generator variation weight used in the tail branch of the
    hand-off (e.g. one member of the sample's 7-point muR/muF variation set).
    `njet`: optional per-event multiplicity of the hard process.  For a multi-jet merged prior,
    pass it to apply the weight only to the 0-jet and 1-jet contributions (see _restrict).
    `rate_tail`: by default the rate factor K is applied only where the calculation is in control
    (the reweighted branch); set True to scale every event, including the generator's tail."""
    d = json.load(open(jpath or _default(energy)))
    K = (d.get('rate', {}).get('K') or 1.0) if rate else 1.0   # null K (rate not certified) -> 1
    return _apply(w0, qT, m_ll, dphi_ll, d['moments'],
                  d['lambda_physical'], d['log_norm_shift'], d['gating'], gate, w0_tail, K,
                  njet, njet_max, rate_tail)


def reweight_scheme(w0, qT, m_ll, dphi_ll, scheme='central',
                    energy="13TeV", jpath=None, gate=True, w0_tail=None, rate=True, njet=None, njet_max=1, rate_tail=False):
    """One scale/NP scheme (scheme='central','2MuR',...). Repeat over all schemes, histogram
    each, and pass the histograms to theory_band() for the band (per-scale quadrature, Wan-Li
    Ju's rule; NOT the envelope over the 29 schemes). With generator tail variations add
    reweight(..., w0_tail=w0_V) for each generator variation V above the hand-off."""
    d = json.load(open(jpath or _default(energy, variations=True)))
    s = d['schemes'][scheme]
    K = (s.get('K') or d.get('rate', {}).get('K') or 1.0) if rate else 1.0
    return _apply(w0, qT, m_ll, dphi_ll, d['moments'],
                  s['lambda_physical'], s['log_norm_shift'], d['gating'], gate, w0_tail, K,
                  njet, njet_max, rate_tail)


def schemes(energy="13TeV", jpath=None):
    """List the available scale/NP scheme names (central + 28 variations)."""
    d = json.load(open(jpath or _default(energy, variations=True)))
    return list(d['schemes'].keys())


# ---------------------------------------------------------------------------------------------------
# The theory band from the 29 scheme histograms: Wan-Li Ju's prescription (his notebook, cell rTSV).
#   up   = sqrt( sum over scales s of  max(h[2s] - h[central], h[0p5s] - h[central], 0)^2 )
#   down = sqrt( sum over scales s of  min(h[2s] - h[central], h[0p5s] - h[central], 0)^2 )
# Each of the 14 scales contributes the larger of its two deviations in each direction, clipped at
# zero (the variations are labelled 0p5 and 2: a factor of two for the scales, the quoted uncertainty ranges for C0_np, kappa_np and MuFtran (Ju, 2026-09-26)); the scales add in quadrature, up and down separately.  The three fixed-order
# members (MuR, MuF, MuRF) are three terms, MuFtran (varied on the resummed and fixed-order sides
# together) one.  This is NOT the envelope over the 28 variations, which keeps only the largest
# scale per bin and is smaller by up to the square root of the number of scales that matter there.
# Identical to the rule of the paper figures (figs/rivet_style.py: scheme_band).  For absolute
# spectra use the per-scheme K (rate.per_scheme) so the rate variation enters as well.
SCALES = ("C0_np", "kappa_np", "MuBeam", "MuSoft", "MuFac", "MuHard", "MuCt", "MuBF",
          "NuBeam", "NuSoft", "MuF", "MuR", "MuRF", "MuFtran")


def scale_of(scheme):
    """'2MuR' -> 'MuR', '0p5MuR' -> 'MuR', 'central' -> None"""
    if scheme.startswith("0p5"): return scheme[3:]
    if scheme.startswith("2"):   return scheme[1:]
    return None


def theory_band(h, central="central"):
    """h: {scheme: array} histograms (or any per-bin quantities) on one binning, keys as in
    schemes().  Returns (lo, hi), the band edges around h[central]."""
    c = np.asarray(h[central], float)
    up2 = np.zeros_like(c); dn2 = np.zeros_like(c); seen = set()
    for s in SCALES:
        devs = [np.asarray(h[k], float) - c for k in h if scale_of(k) == s]
        if not devs: continue
        seen.add(s)
        up2 += np.maximum.reduce([np.maximum(d, 0) for d in devs]) ** 2
        dn2 += np.maximum.reduce([np.maximum(-d, 0) for d in devs]) ** 2
    missing = [k for k in h if k != central and scale_of(k) not in seen]
    if missing: raise ValueError(f"schemes not assigned to a scale: {missing}")
    return c - np.sqrt(dn2), c + np.sqrt(up2)


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
              f"sum(w)/sum(w0)={w.sum()/w0.sum():.4f} (K={d.get('rate',{}).get('K') or 'null -> 1'}), "
              f"above-200-reverts={np.allclose(w[qT>200], w0[qT>200])}, "
              f"n_schemes={len(schemes(energy))}")
