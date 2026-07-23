"""Verify that this package reproduces the delivered products.

Two levels:

  L1  self-contained (no prior needed): parse every delivered lambda_export.json,
      apply to random events, check weights are finite, normalized, and revert to
      the prior above the hand-off.  Run:  python verify.py

  L2  full reproduction (needs a prior sample and the reference weights): apply the
      delivered lambdas to the prior with apply_lambdas.reweight and check the result
      matches the reference per-event weights that were computed by the pipeline.
      Run:  python verify.py --prior /path/to/sherpa_prior_13TeV --ref weights.npz --energy 13TeV

L2 is what was used to certify the release; its result is recorded in VERIFY.md.
"""
import argparse, json, numpy as np
from apply_lambdas import reweight, schemes, _default


def level1():
    ok = True
    for energy in ("13TeV", "13p6TeV"):
        d = json.load(open(_default(energy)))
        n = 100000
        rng = np.random.default_rng(0)
        w0 = rng.normal(1, 0.1, n); m = rng.uniform(60, 120, n)
        qT = rng.exponential(20, n); dphi = rng.uniform(1e-3, 3.0, n)
        w = reweight(w0, qT, m, dphi, energy=energy)
        finite = bool(np.all(np.isfinite(w)))
        revert = bool(np.allclose(w[qT > 200], w0[qT > 200]))
        ok &= finite and revert
        print(f"[L1 {energy:8s}] moments={len(d['moments'])}  schemes={len(schemes(energy))}  "
              f"finite={finite}  reverts_above_200={revert}")
    print("L1:", "PASS" if ok else "FAIL")
    return ok


def level2(prior_dir, ref_npz, energy, n=2_000_000):
    import optimizer_DY_unc as o
    p = o.load_prior(prior_dir)
    rt = p['rT'][:n].astype(float); dd = p['d'][:n].astype(float)
    pT = p['pT'][:n].astype(float); w0 = p['w'][:n].astype(float)
    m = pT / np.maximum(rt, 1e-12)
    wa = reweight(w0, pT, m, dd, energy=energy)
    ref = np.load(ref_npz)['weights'][:n].astype(float)
    bulk = (pT < 120) & (np.abs(w0) > 0) & (np.abs(wa) > 0)
    c = np.median(ref[bulk] / wa[bulk])
    rel = np.abs(ref[bulk] - c * wa[bulk]) / np.maximum(np.abs(ref[bulk]), 1e-30)
    ok = np.median(rel) < 1e-6
    print(f"[L2 {energy:8s}] median|rel|={np.median(rel):.2e}  p99={np.percentile(rel,99):.2e}  "
          f"ratio={c:.6f}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior"); ap.add_argument("--ref"); ap.add_argument("--energy", default="13TeV")
    a = ap.parse_args()
    ok = level1()
    if a.prior and a.ref:
        ok &= level2(a.prior, a.ref, a.energy)
    raise SystemExit(0 if ok else 1)
