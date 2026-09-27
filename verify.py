"""Verify that this package reproduces the delivered products.

Two levels:

  L1  self-contained (no prior needed): parse every delivered lambda_export.json,
      apply to random events and to the corners of phase space (zero acoplanarity, q_T far
      above the hand-off, m_ll at the edge), check that every weight is finite, that the
      weight reverts to the prior above the hand-off, that the multiplicity restriction
      holds on every path, and that bad inputs are refused.  It cannot check the
      normalization, which needs the prior (L2).  Run:  python verify.py

  L2  full reproduction (needs a prior sample and the reference weights): apply the
      delivered lambdas to the prior with apply_lambdas.reweight and require the result to
      equal, without any rescaling, the per-event weights the pipeline wrote (shape, i.e.
      rate=False), and the rate factor K to come out exactly where it is applied.  An error
      in the normalization constant C or in K therefore fails the test.
      Run:  python verify.py --prior /path/to/sherpa_prior_13TeV --ref weights.npz --energy 13TeV

L2 is what was used to certify the release; its result is recorded in VERIFY.md.
"""
import argparse, json, numpy as np
from apply_lambdas import reweight, schemes, _default


def level1():
    ok = True
    for energy in ("13TeV", "13p6TeV", "13TeV_powheg"):
        d = json.load(open(_default(energy)))
        n = 100000
        rng = np.random.default_rng(0)
        w0 = rng.normal(1, 0.1, n); m = rng.uniform(60, 120, n)
        qT = rng.exponential(20, n); dphi = rng.uniform(1e-3, 3.0, n)
        w = reweight(w0, qT, m, dphi, energy=energy)
        finite = bool(np.all(np.isfinite(w)))
        hi = float(d["gating"]["window_GeV"][1])
        gated = d["gating"].get("applied", True) and hi < 1e6
        revert = bool(np.allclose(w[qT > hi], w0[qT > hi])) if gated else True
        # corners: zero and maximal acoplanarity, q_T far beyond the hand-off, m_ll at the edge
        cq = np.array([1200., 150., 150., 0.01, 2000., 50., 199.]); cm = np.array([91., 91., 91., 40., 60., 40., 91.])
        cd = np.array([0., 0., np.pi, 1e-9, 0.5, 3.0, 0.])
        wc = reweight(np.ones(7), cq, cm, cd, energy=energy)
        corners = bool(np.all(np.isfinite(wc))) and (not gated or bool(np.allclose(wc[cq > hi], 1.0)))
        # the multiplicity restriction must hold with and without the hand-off, and keep the tail weight
        wr = reweight(np.ones(2), np.array([50., 50.]), np.array([91., 91.]), np.array([0.3, 0.3]), energy=energy, njet=np.array([2, 0]))
        wg = reweight(np.ones(2), np.array([50., 50.]), np.array([91., 91.]), np.array([0.3, 0.3]), energy=energy, njet=np.array([2, 0]), gate=False)
        wtl = reweight(np.ones(1), np.array([50.]), np.array([91.]), np.array([0.3]), energy=energy, njet=np.array([2]), w0_tail=np.array([2.0]))
        restrict = bool(wr[0] == 1.0 and wg[0] == 1.0 and wtl[0] == 2.0 and wr[1] != 1.0)
        refused = 0
        for bad in ((np.array([1.]), np.array([10.]), np.array([0.]), np.array([0.3])), (np.array([1.]), np.array([10.]), np.array([91.]), np.array([-0.01]))):
            try: reweight(*bad, energy=energy)
            except ValueError: refused += 1
        ok &= finite and revert and corners and restrict and refused == 2
        tail = f"reverts_above_{hi:g}={revert}" if gated else "ungated (pure reweighting everywhere)"
        rb = d.get("rate", {}); K = rb.get("K"); nks = sum(1 for v in rb.get("per_scheme", {}).values() if v.get("K"))
        rate = f"K={K:.4f} ({nks} schemes)" if K else ("rate: sigma_calc shipped, K not set (supply sigma_prior)" if rb else "no rate block")
        print(f"[L1 {energy:12s}] moments={len(d['moments'])}  schemes={len(schemes(energy))}  "
              f"finite={finite}  corners={corners}  restrict={restrict}  bad_inputs_refused={refused == 2}  {tail}  {rate}")
    print("L1:", "PASS" if ok else "FAIL")
    return ok


def level2(prior_dir, ref_npz, energy, n=2_000_000):
    import optimizer_DY_unc as o
    p = o.load_prior(prior_dir)
    rt = p['rT'][:n].astype(float); dd = p['d'][:n].astype(float)
    pT = p['pT'][:n].astype(float); w0 = p['w'][:n].astype(float)
    m = pT / np.maximum(rt, 1e-12)
    ref = np.load(ref_npz)['weights'][:n].astype(float)
    K = json.load(open(_default(energy))).get('rate', {}).get('K') or 1.0
    wa = reweight(w0, pT, m, dd, energy=energy, rate=False)   # the pipeline's per-event weights carry no K
    wk = reweight(w0, pT, m, dd, energy=energy)               # delivered: K on the reweighted branch
    bulk = (pT < 120) & (np.abs(ref) > 1e-30)
    rel = np.abs(ref[bulk] - wa[bulk]) / np.abs(ref[bulk])    # no rescaling: C and the lambdas must be exact
    ok_shape = bool(np.median(rel) < 1e-6 and np.percentile(rel, 99) < 1e-4)
    ratio = float(np.median(wk[bulk] / ref[bulk]))
    ok_rate = abs(ratio - K) < 1e-9
    print(f"[L2 {energy:8s}] n={bulk.sum():,}  median|rel|={np.median(rel):.2e}  p99={np.percentile(rel,99):.2e}  "
          f"max={rel.max():.2e}  K applied/expected={ratio:.6f}/{K:.6f}  {'PASS' if (ok_shape and ok_rate) else 'FAIL'}")
    return ok_shape and ok_rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior"); ap.add_argument("--ref"); ap.add_argument("--energy", default="13TeV")
    a = ap.parse_args()
    ok = level1()
    if a.prior and a.ref:
        ok &= level2(a.prior, a.ref, a.energy)
    raise SystemExit(0 if ok else 1)
