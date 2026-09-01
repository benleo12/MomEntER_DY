"""Statistical admission of candidate moments on a prior -- prior-agnostic, tolerance-free.

    python admit_prune.py <prior_dir> <moments_csv> <pool.json> <out.json>

A constraint <phi_k>_w = t_k +- sigma_k can only be imposed on a sample that can carry it.  Two
tests, both dimensionless comparisons of like with like (no calibrated threshold anywhere):

 (1) PRECISION.  The FIT SAMPLE's estimate of the moment (the seeded 2M-event draw plus the
     extreme tails with their importance factors, exactly the sample every dual fit of the
     pipeline sees -- optimizer_DY_unc.fit_subsample) has a Monte-Carlo error
         sigma_MC,k = sqrt( sum_i (w_i f_i)^2 (phi_ik - mu_k)^2 ) / |sum_i w_i f_i|
     A moment with sigma_MC,k > sigma_stat,k -- the theory statistical error the fit holds it to --
     cannot be fitted from this sample: the constraint lies below the sample's own resolution and
     the optimiser ends up steering the few tail events that carry the moment.  (On the full 51M
     sample the same ratio is < 1 for almost every moment; it is the 2M fit sample that cannot
     resolve the tail-carried rt^2*lnrt^k moments, which is where the fits stall.)  Measured
     2026-08-29 on Sherpa 13 TeV: rt^2*lnrt^4 has max|phi| = 296 feature-sigma while its target
     uncertainty is 4e-8 feature-sigma; Newton then descends the signed-measure chute
     (|lambda| -> 1e3, pulls 1e6, Hessian diag < 0).  These are exactly the moments the old N_eff
     prune removed after hours of thrashing (recipe A: rt^2*lnrt^5, rt^2*lnrt^4, rt^1*lnrt^5, ...),
     and their prior/theory ratios are 0.97-1.05: nothing is lost by not fitting them.

 (2) INFORMATION.  Greedy in order of increasing total degree (sum of exponents; ties: fewer log
     factors first, then name -- pool-independent, so the lowest-order member of a degenerate
     family is the one kept): the standardized column of moment k is projected onto
     the span of the already-admitted columns (|w|-weighted metric, Gram-Schmidt).  The residual
     direction u_k (unit spread) is what the moment adds.  Its target is (t_k - sum_j a_j t_j)/r_k
     with NUMERICAL NOISE  n_k = sqrt(sig_k^2 + sum_j a_j^2 sig_j^2)/r_k  in units of the prior's
     spread along u_k (sig = theory stat (+) sigma_MC: the independent, per-moment noise of the
     targets; the scale variation is a correlated physical shift, not noise, and is propagated by
     the scheme refits).  The penalised dual answers a target change dt_u along u_k with
     lambda_u = dt_u/(r_k^2 + sig_u^2), i.e. the log-weights are modulated across one spread of
     the direction by dt_u r_k/(r_k^2+sig_u^2); for a noise-sized dt_u ~ sig_u that modulation is
     n_k/(1+n_k^2) -- of order ONE when n_k ~ 1.  That is the near-null family problem (same 41
     moments fitted at 13 and 13.6 TeV: lambda corr 0.62, oscillating dphi): the direction is
     real but its target is noise.  A moment is admitted only if n_k <= ADMIT_H (default set from
     the ladder in the sandbox; the noise-driven weight modulation must stay well below the
     few-percent level the reweighting is meant to deliver).  Exact nulls (r_k ~ 1e-13) fail
     automatically; nothing tolerance-like acts on r_k itself.
     The SAME bound is applied to the physical scale variation: for each of the tabulated scheme
     tables s the target shift along u_k, D_s = (dt_s,k - sum_j a_j dt_s,j)/r_k (dt_s = t^s - t,
     computed from the actual correlated tables, no independence assumed), is the log-weight
     modulation that scheme refit will impose across one spread of the direction; along a
     near-null direction it is the physical few-percent shift amplified by 1/r_k.  A moment is
     admitted only if max_s |D_s| <= ADMIT_HS (default = ADMIT_H).  Measured 2026-08-30 without
     it: POWHEG 10M, 20 moments, central agreement 1.3/0.6/2.4% but a dphi scheme band of 17%
     (theory band 2.9%).
 (3) EVENT LEVER.  Bounds (1)-(2) control the log-weight modulation across ONE SPREAD of the new
     direction; a feature with a huge dynamic range (rt^2*lnrt^4 reaches ~300 spreads at rT=50) turns
     a 0.03-spread modulation into e^9 on a handful of events, which then ARE the reweighted
     distribution (measured 2026-08-30: POWHEG, tension-admitted rt^2*lnrt^k, refit dphi band 52%).
     So the WEIGHT SHARE any single event can acquire from the variation budget of one direction
     (central pull + target noise + largest scheme shift, m_i = (|P_u| + n_k + max_s|D_s|) |u_ik|)
     is bounded:   max_i  wa_i (e^{m_i} - 1)  <=  ADMIT_EPS   (default 1e-2: no event may gain more
     than 1% of the fit sample's weight from a 1-sigma budget; measured 2026-08-30 on POWHEG: the moments that broke the refit band sit at 0.05..1e18, the ones that never did at <1e-3
     by ONE event).  Prior-aware: an event's own weight share wa_i enters, so a heavy Sherpa tail
     event is held tighter than a unit-weight POWHEG event.  REPORTED but not enforced by default
     (ADMIT_EPS=inf): the linearised budget on the single most extreme event overstates the real
     (nonlinear, penalised) response by orders of magnitude in the exponent and at any finite
     bound it removed the lnrt^k x dphi^j family the Sherpa fits need (2026-08-30: 13 TeV K 19->12,
     rT 2.0->3.4%).  The scheme refits are guarded downstream instead (final_plots_pro: a scheme
     whose reweighted N_eff collapses is replaced by its damped linear response).

Output: <out.json> with the admitted list (pool order), per-moment diagnostics, and sigma_MC
(normalised units) for the admitted moments; the fits add sigma_MC in quadrature to the stat
penalty (stat (+) prior-MC -- both statistical).  sigma_MC of every moment ever asked for is
cached in <moments_dir>/prior_mc.json, keyed to the prior and the fit-sample definition.
Env: NEV, FIT_NEV (the pipeline's fit-sample definition; defaults 2e8 -> all events, 2e6),
     ADMIT_H (noise-modulation bound).
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimizer_DY_unc as o
from apply_lambdas import features

PRI, CSV, SRC, OUT = sys.argv[1:5]
MOMDIR = os.path.dirname(os.path.abspath(CSV))
NEV = int(os.environ.get('NEV', '200000000')); FIT = int(os.environ.get('FIT_NEV', '2000000')); H = float(os.environ.get('ADMIT_H', '0.1')); HS = float(os.environ.get('ADMIT_HS', str(H))); EPS = float(os.environ.get('ADMIT_EPS', 'inf')); TENSION = int(os.environ.get('ADMIT_TENSION', '0'))
jin = json.load(open(SRC)); names_in = list(jin['selected_moments'])
def _order(nm):
    fs = [f for part in nm.split('×') if part != 'const^0' for f in part.split('*')]
    deg = sum(int(f.split('^')[1]) for f in fs); nlog = sum(1 for f in fs if f.split('^')[0] in ('lnrt', 'lndphi'))
    return (deg, nlog, nm)
names = sorted(names_in, key=_order); K = len(names)
import hashlib; CODE_MD5 = hashlib.md5(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:12]
t0 = time.time()

# ---- theory table (normalised by Z) --------------------------------------------------------
mom = o.load_moments(CSV); mbp = {(a, b): v for a, b, v, u in mom}; mbu = {(a, b): u for a, b, v, u in mom}
ssc = o.compute_sigma_theory(CSV)
ZEROS = ('dphi^0', 'lndphi^0', 'rt^0', 'lnrt^0')
def factors(nm):
    fs = []
    for part in nm.split('×'):
        if part != 'const^0': fs += part.split('*')
    return fs
def lookup(dic, nm):
    fs = factors(nm)
    if len(fs) == 1: cands = [(fs[0], z) for z in ZEROS] + [(z, fs[0]) for z in ZEROS]
    elif len(fs) == 2: cands = [(fs[0], fs[1]), (fs[1], fs[0])]
    else: cands = []
    for k in cands:
        if k in dic and dic[k] is not None: return float(dic[k])
    return None
T = np.array([lookup(mbp, nm) for nm in names], float)
S_stat = np.array([lookup(mbu, nm) for nm in names], float)
S_scale = np.array([lookup(ssc, nm) or 0.0 for nm in names], float)
_cen, _vars = o.get_scale_variations(CSV)
DT = []                                                   # per-scheme target shifts t^s - t (normalised units)
for (fo, res) in _vars:
    _mb = {(a, b): v for a, b, v, u in o.load_moments_for_scale(CSV, fo, res)}
    DT.append(np.array([(lookup(_mb, nm) if lookup(_mb, nm) is not None else T[k]) for k, nm in enumerate(names)], float) - T)
DT = np.array(DT) if DT else np.zeros((0, K)); print(f"admit: {len(DT)} scheme tables loaded for the scale-shift bound")
if not np.all(np.isfinite(T)) or not np.all(np.isfinite(S_stat)):
    bad = [names[k] for k in range(K) if not (np.isfinite(T[k]) and np.isfinite(S_stat[k]))]
    raise SystemExit(f"admit: no theory value/uncertainty for {bad}")

# ---- prior and THE fit sample (same seed/rule as select_stable.py and final_plots_pro.py) ----
p = o.load_prior(PRI); Nf = len(p['w']); NEV = min(NEV, Nf); FIT = min(FIT, NEV)
idx = np.sort(np.random.default_rng(42).choice(Nf, NEV, replace=False))
rt = np.asarray(p['rT'], float)[idx]; d = np.asarray(p['d'], float)[idx]; w = np.asarray(p['w'], float)[idx]; del p
SEL, FAC = o.fit_subsample(rt, d, FIT); wf = w[SEL] * FAC; Wf = float(wf.sum()); NEFF = Wf * Wf / float(np.sum(wf * wf))
print(f"admit: fit sample = {len(SEL):,} events ({int((FAC!=1).sum()):,} random + {int((FAC==1).sum()):,} tail) of NEV={NEV:,}; N_eff(fit)={NEFF:,.0f}")

# ---- (1) MC error of every moment ON THE FIT SAMPLE (cached per prior + sample definition) --
cache = f"{MOMDIR}/prior_mc.json"
mc = json.load(open(cache)) if os.path.exists(cache) else {}
_pr = mc.get('_prior', {}); _key = {'Nf': int(Nf), 'NEV': int(NEV), 'FIT': int(FIT), 'sumw_fit': Wf}
if any(_pr.get(k) != v for k, v in _key.items()):
    mc = {'_prior': {**_key, 'neff_fit': float(NEFF), 'dir': os.path.abspath(PRI),
                     'definition': 'fit sample = seeded NEV draw -> fit_subsample(FIT) with importance factors f; sig_mc = sqrt(sum (w f)^2 (phi-mu)^2)/|sum w f|; lever = max_i |w_i f_i (phi_i-mu)| / |sum w f|'}}
PHI = features(names, rt[SEL], d[SEL])
mu = (wf @ PHI) / Wf
S_mc = np.sqrt(np.maximum(((wf * wf) @ ((PHI - mu)**2)), 0.0)) / abs(Wf)
LEVER = np.abs(wf[:, None] * (PHI - mu)).max(0) / abs(Wf)
for k, nm in enumerate(names): mc[nm] = {'mean': float(mu[k]), 'sig_mc': float(S_mc[k]), 'lever': float(LEVER[k])}
json.dump(mc, open(cache, 'w'), indent=1, ensure_ascii=False)
print(f"admit: sigma_MC on the fit sample for {K} moments [{time.time()-t0:.0f}s]")
ratio1 = S_mc / np.maximum(S_stat, 1e-300)
SHIFT = np.abs(T - mu)                                   # what the fit must move the fit-sample moment by
TENS = SHIFT / np.maximum(S_mc, 1e-300)                  # that shift in units of the sample's own resolution
# (1) admitted if the fit sample resolves the moment to the theory precision (ratio1 <= 1); a moment it
#     cannot hold that tightly is still admitted when the prior-theory tension is significant against
#     the sample's resolution (> 3 sigma_MC) and no single event can supply the shift (lever < shift):
#     the tail towers of a prior that is genuinely off in the tail (POWHEG rt^3..rt^5, |t-mu| = 10-30
#     sigma_MC) are physics the fit must be allowed to correct, held at ~sigma_MC by the penalty.
#     OFF by default (ADMIT_TENSION=1 enables): measured 2026-08-30 on POWHEG, the readmitted
#     rt^2*lnrt^k / rt^3*lndphi^2 gave an excellent central (0.95/0.69/2.5%) but a scheme refit that
#     is one tail event (dphi band 52% against 2.9%); their extreme events sit 1e3-1e6 spreads out.
pass1 = np.logical_or(ratio1 <= 1.0, ((TENS > 3.0) & (LEVER < SHIFT)) if TENSION else np.zeros(K, bool))

# ---- (2) information test on the same fit sample ---------------------------------------------
wa = np.abs(wf) / np.abs(wf).sum()
mu_s = wa @ PHI
X = (PHI - mu_s) * np.sqrt(wa)[:, None]; del PHI
nrm = np.linalg.norm(X, axis=0); nrm[nrm == 0] = 1.0; X /= nrm
sig2 = (S_stat / nrm)**2 + (S_mc / nrm)**2             # independent numerical noise^2 per unit column
sig2sel = sig2 + (S_scale / nrm)**2                     # diagnostic only: with the scale envelope added as if independent
that = (T - mu_s) / nrm                                 # prior->theory pull per unit column (diagnostic only)
RM = np.zeros((K, K)); Q = []; adm = []; rows = []
for k in range(K):
    nm = names[k]
    if not pass1[k]:
        rows.append((nm, 'DROP', f'precision: sigma_MC/sigma_stat={ratio1[k]:.3g} > 1 and tension |t-mu|/sigma_MC={TENS[k]:.2g} (<=3: not resolvable) lever/|t-mu|={LEVER[k]/max(SHIFT[k],1e-300):.2g}', ratio1[k], np.nan, np.nan)); continue
    v = X[:, k].copy(); m = len(Q); c = np.zeros(m)
    for _ in range(2):                                   # modified Gram-Schmidt, one re-orthogonalisation pass
        for i, q in enumerate(Q):
            ci = float(q @ v); v -= ci * q; c[i] += ci
    r = float(np.linalg.norm(v))
    if r < 1e-9: unc = float('inf'); uncsel = float('inf'); a = np.zeros(m)
    else:
        a = np.linalg.solve(RM[:m, :m], c) if m else np.zeros(0)       # x_k = sum_j a_j x_j + r u  (upper-triangular R)
        unc = float(np.sqrt(sig2[k] + np.sum(a * a * sig2[adm])) / r)
        uncsel = float(np.sqrt(sig2sel[k] + np.sum(a * a * sig2sel[adm])) / r)
        dts = (DT[:, k] / nrm[k] - (DT[:, adm] / nrm[adm]) @ a) / r if len(DT) else np.zeros(0)   # per-scheme shift along u_k
        dsmax = float(np.abs(dts).max()) if len(dts) else 0.0
    if r < 1e-9: dsmax = float('inf'); umax = float('inf'); pu = 0.0
    else:
        uabs = np.where(wa > 0, np.abs(v) / (r * np.sqrt(np.maximum(wa, 1e-300))), 0.0)   # |u_i|: event values along u (spread units)
        umax = float(uabs.max())
        pu = float((that[k] - a @ that[adm]) / r) if m else float(that[k] / r)     # central pull along u (spreads)
        _m = np.minimum((abs(pu) + unc + dsmax) * uabs, 60.0)
        evlev = float(np.max(wa * np.expm1(_m)))                                    # largest single-event weight-share gain
    if r < 1e-9: evlev = float('inf')
    if unc <= H and dsmax <= HS and evlev <= EPS:
        Q.append(v / r); RM[:m, m] = c; RM[m, m] = r; adm.append(k)
        rows.append((nm, 'keep', f'sigma_MC/sigma_stat={ratio1[k]:.2g} tension={TENS[k]:.1f}  r={r:.3g}  noise(u)={unc:.3g}  max scheme shift(u)={dsmax:.3g}  event share={evlev:.2g} (umax {umax:.3g})  target(u)={float(that[k]-a@that[adm[:-1]])/r if r>0 else 0:+.2f}', ratio1[k], r, unc))
    else:
        why = (f'information: target noise along the new direction = {unc:.3g} spreads > {H:g}' if unc > H else
               (f'scale: max scheme shift along the new direction = {dsmax:.3g} spreads > {HS:g} (noise {unc:.2g})' if dsmax > HS else
                f'lever: one event would gain {evlev:.2g} of the weight > {EPS:g} ((|pull| {abs(pu):.2g} + noise {unc:.2g} + scheme {dsmax:.2g}) x extreme event {umax:.3g} spreads)'))
        rows.append((nm, 'DROP', why + f' (r={r:.2e}, sigma_MC/sigma_stat={ratio1[k]:.2g})', ratio1[k], r, unc))
kept = [names[k] for k in adm]
# Diagnostic: the dual's curvature is the SIGNED covariance, the admission metric is |w|.  Generalised
# eigenvalues of Cov_signed relative to the |w| Gram of the admitted unit columns: a value <= 0 flags a
# direction the |w| metric certifies but along which the signed measure has no curvature.
if adm:
    Xa = X[:, adm]; wsg = wf / np.abs(wf).sum(); sgn = np.sign(wf)
    Xs = Xa * np.sqrt(np.abs(wsg) / wa)[:, None]           # undo the |w| weighting, apply |w_signed| (same up to norm)
    Cs = (Xs * sgn[:, None]).T @ Xs                         # signed covariance (uncentered about the |w| mean; adequate as a diagnostic)
    Ga = Xa.T @ Xa
    try:
        Lc = np.linalg.cholesky(0.5 * (Ga + Ga.T) + 1e-12 * np.eye(len(adm))); Li = np.linalg.inv(Lc)
        ev = np.linalg.eigvalsh(Li @ (0.5 * (Cs + Cs.T)) @ Li.T); SIGNED_EIG = (float(ev.min()), float(ev.max()))
    except Exception: SIGNED_EIG = (float('nan'), float('nan'))
    print(f"admit: signed-covariance / |w|-Gram generalised eigenvalues of the admitted set: min={SIGNED_EIG[0]:+.3g} max={SIGNED_EIG[1]:.3g}" + ("  !! non-positive direction" if SIGNED_EIG[0] <= 0 else ""))
else: SIGNED_EIG = (float('nan'), float('nan'))
def _fin(x): return None if (x is None or not np.isfinite(x)) else float(x)
n1 = int((~pass1).sum()); n2 = K - n1 - len(kept)
n2b = sum(1 for r_ in rows if r_[1] == 'DROP' and r_[2].startswith('scale')); n2c = sum(1 for r_ in rows if r_[1] == 'DROP' and r_[2].startswith('lever')); n2a = n2 - n2b - n2c
print(f"admit: kept {len(kept)}/{K}  (dropped {n1} below the fit sample's precision, {n2a} with target noise > {H:g}, {n2b} with a scheme shift > {HS:g} spreads along their new direction, {n2c} by the single-event weight-share lever > {EPS:g})  [{time.time()-t0:.0f}s]")
for nm, dec, why, *_ in rows: print(f"  {dec:4s} {nm:34s} {why}")
jin['selected_moments'] = kept; jin['n_selected'] = len(kept)
jin['admit'] = {'rule': f'on the fit sample: (sigma_MC<=sigma_stat OR (|t-mu|>3 sigma_MC AND lever<|t-mu|)) AND numerical target noise (stat(+)MC) along the orthogonal residual direction <= {H:g} prior spreads AND max scheme target shift along it <= {HS:g} spreads AND max single-event weight-share gain from (|pull|+noise+scheme) <= {EPS:g}',
                'H': H, 'HS': HS, 'EPS': EPS, 'n_in': K, 'n_precision_drop': n1, 'n_information_drop': n2, 'fit_sample': int(len(SEL)), 'NEV': int(NEV), 'FIT': int(FIT), 'prior': os.path.abspath(PRI), 'neff_fit': float(NEFF),
                'diag': {nm: {'decision': dec, 'sigma_mc_over_stat': _fin(x), 'r': _fin(rr), 'noise_u': _fin(u)} for nm, dec, why, x, rr, u in rows},
                'order': names, 'code_md5': CODE_MD5, 'signed_eig_min': _fin(SIGNED_EIG[0]), 'signed_eig_max': _fin(SIGNED_EIG[1]),
                'sigma_mc': {nm: float(S_mc[k]) for k, nm in enumerate(names)}}
json.dump(jin, open(OUT, 'w'), indent=1, ensure_ascii=False)
print(f"admit: wrote {OUT}")
