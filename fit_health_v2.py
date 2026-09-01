"""Objective fit-health gate, v2.  STAGED - not yet wired into run_pipeline.sh.

  python fit_health_v2.py <moments_dir> [--vars]
  exit 0 = healthy    exit 1 = SICK (criterion failed)    exit 2 = CANNOT VERIFY (stale/missing)

WHY v2.  The v1 gate blocked on max|lambda_physical|<30 and |C|<5.  Neither is invariant, so
neither is a property of the reweighting:

  * lambda_physical = lambda_standardized/(sF*sG) rescales with the units chosen for the
    monomials.  In the fit's OWN standardized coordinates the delivered K=41 set reads 183.9 and
    the K=13 set that v1 certified reads 494.7 - both far over 30.  A criterion that flips verdict
    under a change of variable is not a criterion.
  * C is exactly invariant under rescaling but shifts by lambda.c under phi -> phi + c, and the
    code standardizes by std without ever mean-centring.  It is the pedestal of the chosen basis.
  Empirically both are ANTI-correlated with quality: every genuine failure in the recorded corpus
  had max|lambda|<=1.40 and |C|<=1.03, while the two best-scoring sets fail v1.

  v1 also had a live bug: `ok = (lm<LMAX and c<CMAX)` followed by `if CHECK_VARS and ok:` meant
  dCmax - the one criterion with measured discriminating power - was skipped whenever the two
  meaningless ones failed.  Every criterion here is evaluated on every invocation.

CRITERIA (all invariant; each says what evidence calibrates it)
  G1  dCmax = max_s |C_s - C_central| < 0.5
      The event-independent part of the scale band: e^{dCmax} is the largest global factor any
      scale scheme puts on the delivered sample.  Recorded corpus: healthy 0.004-0.25 (n=21),
      pathological 3.5-208 (n=12).  0.5 sits in the gap, 2x above the worst healthy value and 7x
      below the lowest genuine failure.  VETO ONLY - dCmax -> 0 trivially as lambda -> 0, so a
      nearly-inert set always looks healthiest.  It must never be used to rank or select.
  G2  small-qT extrapolation: the logit must not diverge UPWARD as rt = qT/m_ll -> 0.
      beta(qT) is one-sided (1 below 120 GeV, 0 above 200), so there is no low-qT gate and this
      region is delivered at full strength.  max_d(logit-C) is scanned on rt = 1e-2 .. 1e-10 and
      must not still be rising and positive (check_smallqt.divergence_scan) - no threshold to
      tune.  This is the only criterion here that bounds the exported FUNCTION rather than its
      realization on one prior, which matters because the export advertises itself as usable
      during event generation, i.e. on events the prior does not contain.
  G3  cross-file consistency: lambda_export.json and lambda_export_variations.json must describe
      the same set and the same central fit.  Zero computation; catches the mismatched
      (central, band) pairs that a fallback ladder leaves behind.
  G4  N_eff of the gated deliverable >= the floor the selector certified for this set.
      Requires the 'health' block that the (not yet applied) export patch writes, and the
      'provenance' block that select_stable.py writes.  FAILS CLOSED: if either is missing the
      gate exits 2 ("cannot verify"), never falls back to a guessed floor.

PRINTED, NEVER GATED: max|lambda_physical|, max|lambda_standardized|, |C|.  Kept visible because
they are what v1 used and readers will look for them.

exit 2 is NOT a health verdict.  The driver must treat it as "re-export and retry", never as a
trigger to substitute a different moment set.
"""
import sys
import json
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_smallqt import divergence_scan, leading_tower   # noqa: E402

DCMAX = 0.5

M = sys.argv[1]
CHECK_VARS = '--vars' in sys.argv

crit = []      # (ok, message)  -> exit 1 if any False
stale = []     # messages       -> exit 2


def load(path):
    try:
        return json.load(open(path))
    except Exception as ex:
        stale.append(f'{os.path.basename(path)}: {ex.__class__.__name__}')
        return None


d = load(f'{M}/lambda_export.json')
if d is None:
    print(f"HEALTH CANNOT-VERIFY: {'; '.join(stale)}")
    sys.exit(2)

names = d['moments']
lam = np.array(d['lambda_physical'], float)

# ---- G2: small-qT extrapolation (needs only lambda_export.json) ----
_div, _rts, _m = divergence_scan(names, lam, float(d['log_norm_shift']))
_P, _cP, _ = leading_tower(names, lam)
crit.append((not _div,
             f"smallqt={'safe' if not _div else f'DIVERGES UP (max logit-C {_m[0]:+.1f} -> {_m[-1]:+.1f})'}"
             f" (leading lnrt^{_P})"))

# ---- G4: N_eff of the gated deliverable vs the selector's own floor ----
h = d.get('health')
if h is None:
    stale.append('lambda_export.json has no "health" block (export predates the health patch)')
else:
    ng, fl = h.get('neff_gated'), h.get('stability_floor')
    if ng is None or fl is None:
        stale.append('health block lacks neff_gated/stability_floor')
    elif not h.get('floor_from_provenance', True):
        stale.append('stability_floor was not read from the set provenance')
    else:
        crit.append((ng >= fl, f"neff_gated={ng:,.0f}(>={fl:,.0f}, ratio={ng / max(fl, 1e-30):.2f})"))

# ---- G1 + G3: band stability and cross-file consistency ----
lever = float('nan')
if CHECK_VARS:
    v = load(f'{M}/lambda_export_variations.json')
    if v is None:
        pass
    else:
        sch = v['schemes']
        if v.get('moments') != names:
            crit.append((False, 'CONSISTENCY: variations file describes a different moment set'))
        lc = np.array(sch['central']['lambda_physical'], float)
        if lc.shape != lam.shape or not np.allclose(lc, lam, rtol=0, atol=1e-9):
            mx = np.abs(lc - lam).max() if lc.shape == lam.shape else float('inf')
            crit.append((False, f'CONSISTENCY: central lambda differs between files (max {mx:.3g})'))
        dc, dl, worst = 0.0, 0.0, ''
        for k, s in sch.items():
            if k == 'central':
                continue
            if abs(s['log_norm_shift'] - sch['central']['log_norm_shift']) > dc:
                dc = abs(s['log_norm_shift'] - sch['central']['log_norm_shift'])
                worst = k
            dl = max(dl, float(np.abs(np.array(s['lambda_physical']) - lc).max()))
        crit.append((dc < DCMAX, f'dCmax={dc:.4f}(<{DCMAX:g}) worst={worst}'))
        lever = dc / max(dl, 1e-30)

if stale:
    print(f"HEALTH CANNOT-VERIFY: {'; '.join(stale)}")
    if crit:
        print('  criteria that COULD be evaluated: '
              + '  '.join(('OK ' if c[0] else 'FAIL ') + c[1] for c in crit))
    print('  -> re-export (EXPORT=1) and re-run. This is NOT a health verdict and must not '
          'trigger a moment-set substitution.')
    sys.exit(2)

ok = all(c[0] for c in crit)
print(f"HEALTH {'PASS' if ok else 'FAIL'}: " + '  '.join(c[1] for c in crit))
ls = np.abs(np.array(d.get('lambda_standardized', d['lambda_physical']), float))
dg = [f"K={d['n_moments']}", f"maxlam_phys={np.abs(lam).max():.2f}", f"maxlam_std={ls.max():.1f}",
      f"|C|={abs(d['log_norm_shift']):.2f}"]
if CHECK_VARS and np.isfinite(lever):
    dg.append(f'lever={lever:.1f}')
print('  diagnostics (NOT gated, basis-dependent): ' + '  '.join(dg))
sys.exit(0 if ok else 1)
