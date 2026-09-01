"""Small-qT extrapolation check for an exported MaxEnt reweighting (prior-agnostic, no events needed).

WHY THIS EXISTS.  The deliverable is  w = w0 * exp( sum_k lam_k phi_k - C ), and the qT gate
beta(qT) is ONE-SIDED: beta=1 below 120 GeV, 0 above 200 GeV.  There is no low-qT gate, so
whatever the fitted function does as qT -> 0 is applied at full strength.  The stability prune
bounds the logit on SAMPLED events; it cannot see a direction in which the function is unbounded
but the prior has (almost) no events.  This check closes that gap, and it needs only the export.

THE TEST (divergence_scan).  Evaluate max over d of (logit - C) on a log-spaced ladder
rt = 1e-2 ... 1e-10 and ask whether it is still rising, positive, and above where it started.
That is a direct statement about the exported FUNCTION: scale-free, offset-free (C cancels in
the comparison of one rt against another), no tunable threshold, and independent of how any one
prior happens to populate the region.

An upward divergence means the reweight factor blows up in the un-gated qT -> 0 region.  A
downward one means the reweighting smoothly switches itself off there, which is benign.

WHY NOT THE ANALYTIC SIGN TEST.  Writing L = ln(rt), every term carrying a positive power of rt
vanishes and what survives is the pure-lnrt tower

    logit(rt,d)  ->  sum_p c_p(d) L^p ,    c_p(d) = sum_{k : rt-side == lnrt^p} lam_k g_k(d)

whose leading power P dominates, so logit -> +inf iff (P odd and c_P<0) or (P even and c_P>0)
-- this is leading_tower()/unsafe_mask() below and it agrees with divergence_scan on every set
measured.  But it is degenerate wherever c_P(d) passes through zero (the delivered K=41 set has
min|c_P| = 2e-6, six orders below its max), and making it well posed needs a negligibility cut,
i.e. a knob.  The numerical scan needs none, so it is the verdict; the tower is kept because it
names the moments responsible, which is what the selector's prune needs in order to drop them.

Usage:  python check_smallqt.py <lambda_export.json> [more.json ...]
Exit 0 if every file is safe over the scanned d range, 1 otherwise.
"""
import json, sys
import numpy as np

DGRID = np.linspace(0.05, 3.10, 200)   # populated range of d = pi - Delta_phi_ll
M_REF = 91.1876                        # only used to quote a qT for the printed table


def parse_moment(nm):
    """'A×B' -> ([(fam,pow)...] rt-side, [(fam,pow)...] dphi-side)."""
    A, B = nm.split('×')

    def side(p):
        if p == 'const^0':
            return []
        return [(s.split('^')[0], int(s.split('^')[1])) for s in p.split('*')]
    return side(A), side(B)


def dphi_side(fl, d):
    out = np.ones_like(d)
    for f, k in fl:
        if k == 0:
            continue
        out = out * (d ** k if f == 'dphi' else np.log(np.maximum(d, 1e-12)) ** k)
    return out


def leading_tower(names, lam, d=DGRID):
    """Return (P, c_P(d), members) for the leading pure-lnrt power."""
    tower, members = {}, {}
    for k, nm in enumerate(names):
        A, B = parse_moment(nm)
        if any(f == 'rt' and p > 0 for f, p in A):
            continue                       # carries rt^m: vanishes as rt -> 0
        P = sum(p for f, p in A if f == 'lnrt')
        tower.setdefault(P, np.zeros_like(d))
        tower[P] = tower[P] + lam[k] * dphi_side(B, d)
        members.setdefault(P, []).append(k)
    P = max(tower)
    return P, tower[P], members[P]


def unsafe_mask(P, cP):
    """True where logit -> +inf as rt -> 0."""
    return (cP < 0) if (P % 2) else (cP > 0)


def logit_on(names, lam, rt, d):
    """logit at scalar rt over the d grid."""
    L = np.log(rt)
    tot = np.zeros_like(d)
    for k, nm in enumerate(names):
        A, B = parse_moment(nm)
        f = 1.0
        for ff, p in A:
            if p == 0:
                continue
            f *= rt ** p if ff == 'rt' else L ** p
        tot = tot + lam[k] * f * dphi_side(B, d)
    return tot


RTS = np.logspace(-2, -10, 9)


def divergence_scan(names, lam, C=0.0, d=DGRID):
    """Direct numerical test: does max_d (logit - C) grow without bound as rt -> 0?

    Preferred over reading the sign of the leading coefficient, which is degenerate wherever
    that coefficient passes through zero and needs a negligibility cut to be well posed.  This
    reads the function itself and needs no threshold: on a log-spaced rt ladder the asymptotic
    polynomial in L=ln(rt) sets in within a couple of decades, so 'still rising at rt=1e-10,
    positive, and above where it started' is an unambiguous statement of unboundedness.
    """
    m = np.array([(logit_on(names, lam, r, d) - C).max() for r in RTS])
    diverges = bool(np.all(np.diff(m[-4:]) > 0) and m[-1] > 0 and m[-1] > m[0])
    return diverges, RTS, m


def check(names, lam, C=0.0, d=DGRID, verbose=True, label=''):
    diverges, rts, m = divergence_scan(names, lam, C, d)
    if verbose:
        P, cP, members = leading_tower(names, lam, d)
        print(f"  leading pure-lnrt power P={P} over {len(members)} moments; "
              f"c_P(d) in [{cP.min():+.5g},{cP.max():+.5g}]")
        verdict = ("UPWARD DIVERGENCE as qT->0 (applied at full strength: beta=1 there)"
                   if diverges else "safe (reweighting is suppressed as qT->0)")
        print(f"  {label}{verdict}")
        print("    max over d of (logit-C):  "
              + "  ".join(f"rt={r:.0e}:{x:+.1f}" for r, x in zip(rts[::2], m[::2])))
        dm = np.array([1.243])             # near the prior mean of d
        for rt in (1e-2, 3e-3, 1e-3, 1e-4):
            x = float(logit_on(names, lam, rt, dm)[0]) - C
            fac = 'overflow' if x > 700 else f"{np.exp(x):.3e}"
            print(f"    rt={rt:<8g} (qT~{rt * M_REF:7.3f} GeV):  logit-C={x:+9.2f}   factor={fac}")
    return (not diverges), m


if __name__ == '__main__':
    allsafe = True
    for path in sys.argv[1:]:
        e = json.load(open(path))
        print(f"\n=== {path}  K={e['n_moments']}  C={e['log_norm_shift']:+.3f}")
        ok, _ = check(e['moments'], np.array(e['lambda_physical']),
                      float(e['log_norm_shift']))
        allsafe &= ok
    sys.exit(0 if allsafe else 1)
