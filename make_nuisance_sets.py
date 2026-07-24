"""Compress the 28 scale schemes into a few nuisance directions.

The per-scheme multiplier shifts Delta-lambda_S = H^{-1} Delta-c_S (linear response,
already stored in lambda_export_variations.json) define the scale covariance in
multiplier space,

    Sigma_lambda = (1/N) sum_S Delta-lambda_S Delta-lambda_S^T ,

equivalent to H^{-1} Sigma_scale H^{-1} in moment space. Its eigendecomposition
gives orthogonal scale nuisance directions. Empirically the spectrum is steep, a
few directions carry essentially all of the scale variance, so the full 28-scheme
band compresses to +-1 sigma multiplier sets along the leading directions,
analogous to Hessian PDF error sets.

Coverage is still quoted from the discrete-scheme envelope. The nuisance sets
provide the correlated-systematic parametrization needed by downstream fits.

  python make_nuisance_sets.py <lambda_export_variations.json> <prior_dir> <out.json> [frac=0.99]

Writes out.json with, per direction: variance fraction, and the +-1 sigma
multiplier sets with their own log-norm shifts (computed on the prior sample so
the applied weights stay normalized and event-local).
"""
import sys, json, numpy as np

def main():
    vpath, prior_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.99
    d = json.load(open(vpath))
    names = d['moments']
    lam_c = np.array(d['schemes']['central']['lambda_physical'])
    D = np.array([np.array(d['schemes'][s]['lambda_physical']) - lam_c
                  for s in d['schemes'] if s != 'central'])
    Sig = D.T @ D / len(D)
    ev, U = np.linalg.eigh(Sig)
    order = np.argsort(ev)[::-1]; ev, U = ev[order], U[:, order]
    cum = np.cumsum(ev) / ev.sum()
    k = int(np.searchsorted(cum, frac) + 1)
    print(f"  {len(D)} schemes, {len(names)} moments; top {k} directions carry "
          f"{100*cum[k-1]:.1f}% of scale variance "
          f"({', '.join(f'{100*e/ev.sum():.0f}%' for e in ev[:k])})")

    # log-norm shift per multiplier set, computed on the prior (signed weights)
    from apply_lambdas import features
    import pandas as pd, os
    dphi = pd.read_csv(os.path.join(prior_dir, 'dphi_values.csv.gz')).values.ravel()
    pT   = pd.read_csv(os.path.join(prior_dir, 'pT_values.csv.gz')).values.ravel()
    m    = pd.read_csv(os.path.join(prior_dir, 'm_values.csv.gz')).values.ravel()
    try:    w0 = pd.read_csv(os.path.join(prior_dir, 'pT_weight.csv.gz')).values.ravel()
    except Exception: w0 = np.ones_like(pT)
    N = min(len(pT), 2_000_000)
    rt = (pT[:N]/m[:N]).astype(float); aco = (np.pi - dphi[:N]).astype(float)
    w0 = w0[:N].astype(float)
    F = features(names, rt, aco)
    def shift(lam):
        logit = F @ lam
        g = logit.max()
        return float(g + np.log((w0*np.exp(logit-g)).sum()/w0.sum()))

    dirs = []
    for a in range(k):
        step = np.sqrt(ev[a]) * U[:, a]
        lam_p, lam_m = lam_c + step, lam_c - step
        dirs.append({'variance_fraction': float(ev[a]/ev.sum()),
                     'lambda_physical_up':   list(lam_p), 'log_norm_shift_up':   shift(lam_p),
                     'lambda_physical_down': list(lam_m), 'log_norm_shift_down': shift(lam_m)})
    json.dump({'description': ('Scale nuisance directions: eigendecomposition of the '
               '28-scheme multiplier covariance. Apply like any scheme: '
               'w = w0*exp(sum lambda_k phi_k - log_norm_shift), then the standard gating. '
               'Coverage is quoted from the scheme envelope, these sets provide the '
               'correlated parametrization for downstream fits.'),
               'moments': names, 'n_directions': k,
               'variance_covered': float(cum[k-1]), 'directions': dirs},
              open(out, 'w'))
    print(f"  wrote {out} ({k} directions, +-1 sigma sets with normalization shifts)")

if __name__ == '__main__':
    main()
