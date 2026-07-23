#!/usr/bin/env python3
"""
MaxEnt Reweighting — Penalized Dual with Newton's Method

Minimizes the convex penalized dual:

    L(λ) = log Z(λ) − Σ_k λ_k μ_k + ½ Σ_k σ_k² λ_k²

where Z(λ) = Σ_i w_i^0 exp(Σ_k λ_k g_k(x_i))

Gradient:  ∂L/∂λ_k = ⟨g_k⟩_w − μ_k + σ_k² λ_k
Hessian:   ∂²L/∂λ_j∂λ_k = Cov_w(g_j, g_k) + σ_k² δ_{jk}

Both are cheap for ~27 constraints: 27×27 system solved each Newton step.
The dual is strictly convex ⇒ guaranteed convergence, no staging needed.

Changes from χ² version:
  1. Penalized dual loss (convex) replaces χ² (non-convex)
  2. Newton's method replaces L-BFGS/ADAM
  3. Feature winsorization caps extreme tails
"""

import os
# Cap BLAS/OpenMP threads to 1 BEFORE importing numpy. We parallelize across
# candidates with our own worker pool; letting numpy/Accelerate also spawn
# per-op threads causes catastrophic oversubscription (load avg 100s) and
# extreme slowdowns. VECLIB_MAXIMUM_THREADS is the macOS Accelerate knob.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEVICE = "cpu"  # Newton on 27×27 doesn't need GPU

SELECTED_MOMENT_NAMES = None  # Use --moments_file instead
# SELECTED_MOMENT_NAMES = {
#     # rT marginals
#     "rt^1×const^0",
#     "lnrt^1×const^0",
#     "lnrt^2×const^0",
#     "lnrt^3×const^0",
#     # dphi marginals
#     "const^0×dphi^1",
#     "const^0×dphi^2",
#     "const^0×dphi^3",
#     "const^0×lndphi^1",
#     "const^0×lndphi^2",
#     "const^0×lndphi^3",
#     # cross rT×dphi
#     "rt^1×dphi^1",
#     "rt^1×dphi^2",
#     "lnrt^1×dphi^1",
#     "lnrt^1×dphi^2",
#     "lnrt^2×dphi^1",
#     "lnrt^2×dphi^2",
#     "lnrt^3×dphi^1",
#     "rt^1×lndphi^1",
#     "rt^1×lndphi^2",
#     "lnrt^1×lndphi^1",
#     "lnrt^1×lndphi^2",
#     "lnrt^2×lndphi^1",
#     "lnrt^2×lndphi^2",
#     # composites — only rt^1
#     "lnrt^1×rt^1",
#     "lnrt^2×rt^1",
#     "lnrt^3×rt^1",
# }


# ========================================
# Configuration
# ========================================
def get_args():
    p = argparse.ArgumentParser(
        description="""MaxEnt Reweighting with two modes:
  select: Find best moments using greedy TD selection on a subset of events.
          Saves selected moments to a JSON file.
  run:    Fit reweighting using selected moments (from JSON) on all events.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mode
    p.add_argument("--mode", choices=["select", "run", "plot"], default="run",
                    help="'select': find best moments (fast, subset). "
                         "'run': fit with selected moments (full dataset). "
                         "'plot': replot from saved lambdas (no Newton solve).")

    # Data paths
    p.add_argument("--base_dir", default="/Users/user/Library/CloudStorage/Dropbox/LogMoments/LogdPhipT/ps")
    p.add_argument("--prior_dir", default=None)
    p.add_argument("--mom_dir", default=None)
    p.add_argument("--accs", nargs="+", default=["N2LL'+NNLO"])
    p.add_argument("--reweight_variations", action="store_true",
                    help="Reweight all scale variations (run mode)")
    p.add_argument("--rebin_factor", type=int, default=3,
                    help="Rebin factor for plot histograms (default: 3, i.e. 80→26 bins)")

    # Shared optimization
    p.add_argument("--max_newton_steps", type=int, default=50,
                    help="Max Newton iterations")
    p.add_argument("--newton_tol", type=float, default=1e-9,
                    help="Convergence tolerance on max |gradient|")
    p.add_argument("--winsorize_pct", type=float, default=99.99,
                    help="Percentile for feature winsorization (0=off)")
    p.add_argument("--max_events", type=int, default=None,
                    help="Cap total events (for testing)")

    # Select mode options
    p.add_argument("--select_max_moments", type=int, default=30,
                    help="Max moments to select (select mode)")
    p.add_argument("--select_n_events", type=int, default=1_000_000,
                    help="Events for selection trials (select mode, default: 1M). "
                         "Selection only needs to rank candidates reliably; "
                         "full statistics are used in run mode.")
    p.add_argument("--select_max_k_rt", type=int, default=5,
                    help="Max power of rT features in candidate pool (select mode, default: 5)")
    p.add_argument("--select_max_k_dphi", type=int, default=5,
                    help="Max power of dphi features in candidate pool (select mode, default: 5)")
    p.add_argument("--max_pair_power", type=int, default=6,
                    help="Maximum total power of a candidate moment pair "
                         "(default: 6). Prevents high-variance cross-products "
                         "like rt^5×lndphi^4 (power 9) while allowing "
                         "tail-sensitive marginal moments like rt^5×const^0 (power 5).")
    p.add_argument("--n_workers", type=int, default=1,
                    help="Parallel workers for screening (select mode). "
                         "Set to number of CPU cores, e.g. 8.")
    p.add_argument("--select_min_improvement", type=float, default=0.1,
                    help="Stop selection when TD improvement < this %% (default: 0.5)")
    p.add_argument("--select_min_steps", type=int, default=5,
                    help="Minimum selection steps before TD-based stopping (default: 5). "
                         "Allows TD to worsen initially while building up moments.")
    p.add_argument("--screen_top_k", type=int, default=10000,
                    help="Candidates passed from fast screening to full Newton eval "
                         "(default: 10000 = all candidates). Reduce only if runtime "
                         "is prohibitive and you have verified the screener is reliable.")
    p.add_argument("--screen_steps", type=int, default=4,
                    help="Newton steps used for fast screening stage (default: 2).")
    p.add_argument("--max_component_worsen", type=float, default=None,
                    help="Maximum allowed worsening of either TD component relative "
                         "to its running best, as a fraction (default: None = disabled). "
                         "E.g. 0.1 means if rT TD reached 9000 at its best, no subsequent step "
                         "may push it above 9900. Prevents trading away a well-corrected "
                         "observable. Falls back to unconstrained if all candidates violate.")

    # Run mode options
    p.add_argument("--moments_file", default=None,
                    help="JSON file with selected moments (from select mode). "
                         "If not provided, uses SELECTED_MOMENT_NAMES in code.")
    p.add_argument("--event_stages", type=int, default=1,
                    help="Number of event doubling stages for run mode")
    p.add_argument("--stat_band", action="store_true",
                    help="Fit target±σ_stat to get statistical uncertainty band")
    p.add_argument("--regularize", action="store_true",
                    help="Use target σ as regularization in select and run modes (default: σ=0)")

    # Plot mode options
    p.add_argument("--lambdas_json", default=None,
                    help="Path to lambdas_all_variations JSON (plot mode). "
                         "If not provided, looks in output_dir.")

    # Output
    p.add_argument("--output_dir", default=None)
    p.add_argument("--hist_csv", default=None,
                    help="Path to histogram CSV with target distributions")
    p.add_argument("--pT_theory_file", default=None,
                    help="Path to Wan-Li qT_1D_Dist .m theory file. If given, the "
                         "pT plot includes theory qT + Resum/FO/C0/kappa sub-panels "
                         "(matching the rT/dphi multi-panel style).")
    p.add_argument("--preselect", default=None,
                    help="Comma-separated moment names to force-include before greedy select "
                         "(e.g. \"const^0×dphi^1,const^0×lndphi^1\"). "
                         "Forced moments are never removed by the backward step.")
    p.add_argument("--dist_marginals", action="store_true",
                    help="Replace same-type marginal targets with values computed from "
                         "binned distributions (ensures consistency with TD target)")
    p.add_argument("--no_backward", action="store_true",
                    help="Disable backward removal step in greedy selection (faster)")
    p.add_argument("--select_strategy", default="total",
                    choices=["total", "alternating", "rT_then_dphi", "chi2", "chi2_phased", "td_phased"],
                    help="Selection strategy: 'total' picks by total TD (default), "
                         "'alternating' picks by TD_rT on odd steps and TD_dphi on even steps, "
                         "'rT_then_dphi' optimizes rT marginals first, then dphi marginals, "
                         "then cross-terms, 'chi2' picks by chi2/bin (tail-sensitive).")
    p.add_argument("--binned_select", action="store_true",
                    help="Select on binned event surrogate (~1000× faster). "
                         "Bins events into theory's rT×dphi grid; uses bin "
                         "centers as effective events with summed weights.")
    p.add_argument("--is_select", action="store_true",
                    help="Importance-sampled subsample for selection: pick "
                         "N' events with probability ∝ |w_i|, replicate "
                         "selected events (with original signed weights) so "
                         "moment estimators are unbiased.")
    p.add_argument("--gpu", action="store_true",
                    help="Use PyTorch GPU backend (MPS on Mac, CUDA on Linux). "
                         "Enables ~10-60× speedup for selection on large priors.")
    p.add_argument("--phase1_pure_rt", action="store_true",
                    help="In chi2_phased Phase 1, restrict candidates to pure-rT "
                         "moments (exclude cross terms). Cross terms become "
                         "available in Phase 2 alongside dphi moments.")
    p.add_argument("--rebin_atlas", action="store_true",
                    help="Rebin target rTDist and dphiDist onto ATLAS-style "
                         "variable-width bins (~18 rT bins, 13 dphi bins) for "
                         "chi2 selection. Aggregates 500/80 native bins so chi2 "
                         "weighting matches the peak/tail structure shown in plots.")
    p.add_argument("--rebin_log", action="store_true",
                    help="Rebin target rTDist (only) onto log-uniform bins "
                         "(equal bins per decade). dphi stays at native binning.")
    p.add_argument("--td_log_density", action="store_true",
                    help="Compute TD_rT on log-densities: TD(log p, log q) instead of "
                         "TD(p, q). Treats relative deviations equally across decades. "
                         "Recommended with --rebin_log.")
    p.add_argument("--coarse_bins", action="store_true",
                    help="Use coarser ATLAS/log bin variants (suitable for low-statistics "
                         "runs ~1M events). Cuts bin count roughly in half.")
    p.add_argument("--target_rel_acc", type=float, default=0.0,
                    help="If >0, override target σ_unc with rel_acc * t_cen for chi² "
                         "computation. Drives metric toward 'all bins within rel_acc%%'. "
                         "Recommended with chi2_phased + --rebin_log (e.g., 0.02 for 2%%).")
    p.add_argument("--reg_linear", action="store_true",
                    help="Use σ (linear) instead of σ² in the dual regularizer. "
                         "Regularizes high-σ (tail) moments less harshly than the "
                         "Bayesian σ² default.")
    p.add_argument("--sigma_theory", action="store_true",
                    help="Use the THEORY (scale-variation envelope) uncertainty as "
                         "the regularizer σ instead of the statistical/numerical "
                         "σ. Per moment, σ_theory = max_v|μ_v−μ_central| over scale "
                         "variations, floored at the statistical σ.")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    args.prior_dir = args.prior_dir or f"{args.base_dir}/sherpa_prior_1"
    args.mom_dir = args.mom_dir or f"{args.base_dir}/moments_8TeV"
    args.output_dir = args.output_dir or f"{args.base_dir}/maxent_output"
    os.makedirs(args.output_dir, exist_ok=True)
    # Propagate global flags for compute_td_split / MaxEntDual
    global _TD_LOG_DENSITY, _TARGET_REL_ACC, _REG_LINEAR
    _TD_LOG_DENSITY = getattr(args, 'td_log_density', False)
    _TARGET_REL_ACC = float(getattr(args, 'target_rel_acc', 0.0))
    _REG_LINEAR = getattr(args, 'reg_linear', False)
    return args


def triangle_divergence(p, q, eps=1e-12):
    """Triangle divergence TD(p||q) = Σ (p_i - q_i)^2 / (p_i + q_i)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = np.isfinite(p) & np.isfinite(q) & ((p + q) > 0)
    p = p[mask]; q = q[mask]
    return np.sum((p - q)**2 / (p + q + eps))


def chi2_divergence(p, q, eps=1e-12):
    """χ² divergence: Σ (p_i - q_i)^2 / q_i.  Penalizes tails heavily."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = np.isfinite(p) & np.isfinite(q) & (q > eps)
    p = p[mask]; q = q[mask]
    return np.sum((p - q)**2 / q)


def chi2_per_bin(p, q, q_unc, eps=1e-30):
    """Per-bin χ²: (1/N_bins) Σ (p_i - q_i)^2 / σ_i^2.

    Uses target uncertainty σ_i per bin. Bins with zero unc are skipped.
    Returns (chi2_per_bin, n_bins_used).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    q_unc = np.asarray(q_unc, dtype=float)
    mask = (np.isfinite(p) & np.isfinite(q) & np.isfinite(q_unc)
            & (q_unc > eps) & (q > 0))
    if mask.sum() == 0:
        return 0.0, 0
    p = p[mask]; q = q[mask]; q_unc = q_unc[mask]
    chi2 = np.sum((p - q)**2 / q_unc**2)
    n_bins = len(p)
    return chi2 / n_bins, n_bins


# ========================================
# Load Prior Data
# ========================================
def load_prior(prior_dir):
    """Load prior MC events"""
    print(f"\n[Loading Prior from {prior_dir}]")

    dphi = pd.read_csv(f"{prior_dir}/dphi_values.csv.gz").values.flatten()
    pT = pd.read_csv(f"{prior_dir}/pT_values.csv.gz").values.flatten()
    m = pd.read_csv(f"{prior_dir}/m_values.csv.gz").values.flatten()

    n = min(len(dphi), len(pT), len(m))
    dphi, pT, m = dphi[:n], pT[:n], m[:n]

    try:
        w_pT = pd.read_csv(f"{prior_dir}/pT_weight.csv.gz").values.flatten()[:n]
        w = w_pT.astype(np.float64)
    except Exception:
        w = np.ones(n, dtype=np.float64)

    good = np.isfinite(m) & (m > 1e-300)
    dphi = dphi[good]
    pT = pT[good]
    m = m[good]
    w = w[good]

    print(f"  Total events: {n:,}")
    print(f"  Good events after filtering: {len(dphi):,} ({100*len(dphi)/n:.2f}%)")

    d = (np.pi - dphi).astype(np.float64)
    rT = (pT / m).astype(np.float64)

    print(f"  d=(π-Δφ): mean={d.mean():.4f}, range=[{d.min():.4f}, {d.max():.4f}]")
    print(f"  rT=pT/m: mean={rT.mean():.4f}, range=[{rT.min():.4f}, {rT.max():.4f}]")

    return {"d": d, "rT": rT, "pT": pT, "m": m, "w": w}


# ========================================
# Load Target Moments (WITH UNCERTAINTIES)
# ========================================
def parse_moment(s):
    """Parse 'rt^1' or '(lnrt)^2' → (base, power)"""
    s = s.strip().lower()
    is_log = 'ln' in s

    if 'rt' in s:
        base = 'lnrt' if is_log else 'rt'
    elif 'dphi' in s:
        base = 'lndphi' if is_log else 'dphi'
    else:
        return None, 0

    m = re.search(r'\^(\d+)', s)
    k = int(m.group(1)) if m else 1
    return base, k


def load_moments(mom_path):
    """Load central moments AND uncertainties from CSV"""
    print(f"\n[Loading Moments from {os.path.basename(mom_path)}]")

    df = pd.read_csv(mom_path)

    # Find columns (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    c_fo = cols.get('scalefo') or cols.get('fo')
    c_res = cols.get('scaleres') or cols.get('res')
    c_o1 = cols['o1']
    c_o2 = cols['o2']

    # Find value column
    value_col = None
    for name in ['value', 'val', 'moment']:
        if name in cols:
            value_col = cols[name]
            break
    if not value_col:
        value_col = df.columns[-2] if 'uncertainty' in cols else df.columns[-1]

    # Check for uncertainty column
    has_unc = 'uncertainty' in cols
    unc_col = cols.get('uncertainty')
    if has_unc:
        print(f"  Found uncertainty column: {unc_col}")

    # Filter central rows
    central = df[
        df[c_fo].str.contains('CV', case=False, na=False) &
        df[c_res].str.contains('CV', case=False, na=False)
    ].copy()

    if central.empty:
        raise RuntimeError(f"No central rows found in {mom_path}")

    # Extract moments with uncertainties
    moments = []
    Z, Z_unc = None, None

    for _, row in central.iterrows():
        o1, o2 = str(row[c_o1]), str(row[c_o2])
        val = float(row[value_col])
        unc = float(row[unc_col]) if has_unc else None

        if 'rt^0' in o1.lower() and 'dphi^0' in o2.lower():
            Z = val
            Z_unc = unc

        moments.append((o1, o2, val, unc))

    if Z is None:
        raise RuntimeError("Normalization moment (rt^0 × dphi^0) not found")

    # Normalize values AND propagate uncertainties
    moments_norm = []
    for o1, o2, val, unc in moments:
        val_norm = val / Z

        if unc is not None and Z_unc is not None and val != 0:
            rel_unc_val = unc / abs(val)
            rel_unc_Z = Z_unc / Z
            unc_norm = abs(val_norm) * np.sqrt(rel_unc_val**2 + rel_unc_Z**2)
        else:
            unc_norm = None

        moments_norm.append((o1, o2, val_norm, unc_norm))

    print(f"  Central moments: {len(moments_norm)}")
    print(f"  Normalization: {Z:.6e} ± {Z_unc:.6e}" if Z_unc else f"  Normalization: {Z:.6e}")

    uncs = [u for _, _, _, u in moments_norm if u is not None]
    if uncs:
        vals = [v for _, _, v, u in moments_norm if u is not None]
        rel_uncs = [100*u/abs(v) if v != 0 else 0 for v, u in zip(vals, uncs)]
        print(f"  Relative uncertainties: min={min(rel_uncs):.2f}%, median={np.median(rel_uncs):.2f}%, max={max(rel_uncs):.2f}%")

    return moments_norm


def compute_composite_moments_from_distributions(hist_csv, acc, max_k=3, needed_pairs=None):
    """
    Compute same-type composite moments (rT×rT, dphi×dphi) from binned distributions.
    """
    import csv
    from collections import defaultdict

    if hist_csv is None:
        return []

    if isinstance(hist_csv, (list, tuple)):
        csv_files = list(hist_csv)
    else:
        csv_files = [hist_csv]

    rows = []
    for path in csv_files:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = {
                        'dist': row['dist'],
                        'acc': row['acc'],
                        'fo': row['ScaleFO'],
                        'res': row['ScaleRes'],
                        'bin_lo': float(row['bin_lo']),
                        'bin_hi': float(row['bin_hi']),
                        'density': float(row.get('density', 'nan')),
                        'uncertainty': float(row.get('uncertainty', 'nan')) if 'uncertainty' in row else np.nan,
                    }
                    rows.append(r)
                except Exception:
                    continue

    if not rows:
        return []

    acc_variants = {acc, acc.replace("'", "p"), acc.replace("'", ""),
                     acc.replace("p", "'")}
    acc_rows = [r for r in rows if r['acc'] in acc_variants]
    if not acc_rows:
        return []

    by_dist = defaultdict(list)
    for r in acc_rows:
        by_dist[r['dist']].append(r)

    dist_config = {
        'rTDist': {'var': 'rt', 'logvar': 'lnrt'},
        'dphiDist': {'var': 'dphi', 'logvar': 'lndphi'},
    }

    composite_moments = []

    for dist_name, config in dist_config.items():
        if dist_name not in by_dist:
            continue

        dist_rows = by_dist[dist_name]
        by_scale = defaultdict(list)
        for r in dist_rows:
            by_scale[(r['fo'], r['res'])].append(r)

        central_key = None
        for key in by_scale:
            if 'CV' in str(key[0]).upper() and 'CV' in str(key[1]).upper():
                central_key = key
                break
        if central_key is None:
            continue

        central_rows = sorted(by_scale[central_key], key=lambda r: r['bin_lo'])
        edges = np.array([central_rows[0]['bin_lo']] + [r['bin_hi'] for r in central_rows])
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        densities = np.array([r['density'] for r in central_rows])
        uncertainties = np.array([r['uncertainty'] if np.isfinite(r['uncertainty']) else 0.0
                                  for r in central_rows])

        Z_dist = np.sum(densities * widths)
        safe_centers = np.maximum(centers, 1e-30)
        ln_centers = np.log(safe_centers)

        var_name = config['var']
        logvar_name = config['logvar']

        for j in range(1, max_k + 1):
            for k in range(1, max_k + 1):
                o1 = f"{logvar_name}^{k}"
                o2 = f"{var_name}^{j}"

                if needed_pairs is not None:
                    name_fwd = _normalize_moment_name(f"{o1}×{o2}")
                    name_rev = _normalize_moment_name(f"{o2}×{o1}")
                    ckey = _composite_key(logvar_name, k, var_name, j)
                    name_comp = _normalize_moment_name(ckey.replace('*', '×'))
                    if not (name_fwd in needed_pairs or name_rev in needed_pairs or name_comp in needed_pairs):
                        continue

                f_vals = (safe_centers ** j) * (ln_centers ** k)
                integrand = f_vals * densities * widths
                val = np.sum(integrand) / Z_dist

                deriv = (f_vals - val) * widths / Z_dist
                unc_stat = np.sqrt(np.sum((deriv * uncertainties)**2))
                unc_sys = 0.01 * abs(val)
                unc = np.sqrt(unc_stat**2 + unc_sys**2)

                composite_moments.append((o1, o2, val, unc if unc > 0 else None))

    if composite_moments:
        print(f"\n[Composite Moments from Binned Distributions]")
        print(f"  Computed {len(composite_moments)} composite moments")
        for o1, o2, val, unc in composite_moments:
            rel = 100*unc/abs(val) if unc and val != 0 else 0
            print(f"    {o1}×{o2}: {val:.6e} ± {unc:.3e} ({rel:.1f}%)" if unc else f"    {o1}×{o2}: {val:.6e}")

    return composite_moments


def get_scale_variations(mom_path):
    """Get all scale variations in the moments file"""
    df = pd.read_csv(mom_path)
    cols = {c.lower(): c for c in df.columns}
    c_fo = cols.get('scalefo') or cols.get('fo')
    c_res = cols.get('scaleres') or cols.get('res')

    variations = df[[c_fo, c_res]].drop_duplicates().values.tolist()

    central = None
    others = []
    for fo, res in variations:
        if 'CV' in str(fo).upper() and 'CV' in str(res).upper():
            central = (fo, res)
        else:
            others.append((fo, res))

    return central, others


def load_moments_for_scale(mom_path, scale_fo, scale_res):
    """Load moments (with uncertainties) for a specific scale variation"""
    df = pd.read_csv(mom_path)

    cols = {c.lower(): c for c in df.columns}
    c_fo = cols.get('scalefo') or cols.get('fo')
    c_res = cols.get('scaleres') or cols.get('res')
    c_o1, c_o2 = cols['o1'], cols['o2']

    value_col = None
    for name in ['value', 'val', 'moment']:
        if name in cols:
            value_col = cols[name]
            break
    if not value_col:
        value_col = df.columns[-2] if 'uncertainty' in cols else df.columns[-1]

    has_unc = 'uncertainty' in cols
    unc_col = cols.get('uncertainty')

    selected = df[(df[c_fo] == scale_fo) & (df[c_res] == scale_res)].copy()
    if selected.empty:
        return None

    moments, Z, Z_unc = [], None, None
    for _, row in selected.iterrows():
        o1, o2 = str(row[c_o1]), str(row[c_o2])
        val = float(row[value_col])
        unc = float(row[unc_col]) if has_unc else None

        if 'rt^0' in o1.lower() and 'dphi^0' in o2.lower():
            Z = val
            Z_unc = unc
        moments.append((o1, o2, val, unc))

    if Z is None:
        return None

    moments_norm = []
    for o1, o2, val, unc in moments:
        val_norm = val / Z
        if unc is not None and Z_unc is not None and val != 0:
            rel_unc_val = unc / abs(val)
            rel_unc_Z = Z_unc / Z
            unc_norm = abs(val_norm) * np.sqrt(rel_unc_val**2 + rel_unc_Z**2)
        else:
            unc_norm = None
        moments_norm.append((o1, o2, val_norm, unc_norm))

    return moments_norm


def compute_sigma_theory(mom_path):
    """Per-moment THEORY (scale) uncertainty from the scale-variation envelope.

    For each scale combo the moment is normalized by THAT combo's own Z
    (rt^0×dphi^0), then σ_theory(o1,o2) = max_v |μ_v − μ_central| over the
    non-central variations. Returns {(o1,o2): σ_theory_norm}. Used to replace
    the statistical/numerical regularizer σ with the theory scale uncertainty.
    """
    df = pd.read_csv(mom_path)
    cols = {c.lower(): c for c in df.columns}
    c_fo = cols.get('scalefo') or cols.get('fo')
    c_res = cols.get('scaleres') or cols.get('res')
    c_o1, c_o2 = cols['o1'], cols['o2']
    value_col = None
    for name in ['value', 'val', 'moment']:
        if name in cols:
            value_col = cols[name]
            break
    if not value_col:
        value_col = df.columns[-2] if 'uncertainty' in cols else df.columns[-1]

    norm = {}  # (fo,res) -> {(o1,o2): val_norm}
    for (fo, res), grp in df.groupby([c_fo, c_res]):
        d = {(str(r[c_o1]), str(r[c_o2])): float(r[value_col])
             for _, r in grp.iterrows()}
        Z = None
        for (o1, o2), v in d.items():
            if 'rt^0' in o1.lower() and 'dphi^0' in o2.lower():
                Z = v
                break
        if not Z:
            continue
        norm[(fo, res)] = {k: v / Z for k, v in d.items()}

    cen_key = None
    for (fo, res) in norm:
        if 'CV' in str(fo).upper() and 'CV' in str(res).upper():
            cen_key = (fo, res)
            break
    if cen_key is None:
        return {}
    cen = norm[cen_key]

    sigma = {}
    for key, mu0 in cen.items():
        devs = [abs(norm[k].get(key, mu0) - mu0) for k in norm if k != cen_key]
        sigma[key] = max(devs) if devs else 0.0
    return sigma


# ========================================
# Build Features
# ========================================
def _is_rt_type(base):
    return base in ('rt', 'lnrt')

def _is_dphi_type(base):
    return base in ('dphi', 'lndphi')

def _compute_basis(values, base, k):
    if k == 0:
        return np.ones(len(values), dtype=np.float64)
    if base in ('rt', 'dphi'):
        return values ** k
    else:
        return np.log(np.maximum(values, 1e-30)) ** k

def _composite_key(b1, k1, b2, k2):
    parts = sorted([(b1, k1), (b2, k2)])
    return f"{parts[0][0]}^{parts[0][1]}*{parts[1][0]}^{parts[1][1]}"

def _display_name(f_name, g_name):
    fb, fk = f_name
    gb, gk = g_name
    is_f_composite = isinstance(fb, str) and '*' in fb
    is_g_composite = isinstance(gb, str) and '*' in gb
    if is_f_composite and gb == 'const' and gk == 0:
        return fb.replace('*', '×')
    elif is_g_composite and fb == 'const' and fk == 0:
        return gb.replace('*', '×')
    else:
        return f"{fb}^{fk}×{gb}^{gk}"

def _normalize_moment_name(name):
    parts = name.split('×')
    if len(parts) == 2:
        return '×'.join(sorted(parts))
    return name


def build_features(prior, moments, max_k_rt, max_k_dphi, winsorize_pct=99.9):
    """Build F (rT) and G (dphi) feature matrices with optional winsorization.

    If `prior` was made by make_binned_prior(), uses the EXACT per-cell
    moments (rT_pow[k], lnrT_pow[k], etc.) instead of (mean rT)^k.
    """
    print(f"\n[Building Features: max_k_rt={max_k_rt}, max_k_dphi={max_k_dphi}]")

    rT = prior['rT'].astype(np.float64)
    d = prior['d'].astype(np.float64)
    binned = bool(prior.get('_binned', False))
    if binned:
        print(f"  [binned mode: using exact per-cell moments]")
        rT_pow_cell   = prior['rT_pow']
        lnrT_pow_cell = prior['lnrT_pow']
        d_pow_cell    = prior['dphi_pow']
        lnd_pow_cell  = prior['lndphi_pow']

    # ── Winsorize extreme values (skip for binned: cell means already smoothed) ──
    if not binned and winsorize_pct > 0 and winsorize_pct < 100:
        rT_cap = np.percentile(rT, winsorize_pct)
        d_cap = np.percentile(d, winsorize_pct)
        n_rT_capped = np.sum(rT > rT_cap)
        n_d_capped = np.sum(d > d_cap)
        rT = np.minimum(rT, rT_cap)
        d = np.minimum(d, d_cap)
        print(f"  Winsorization at {winsorize_pct}th percentile:")
        print(f"    rT capped at {rT_cap:.4f} ({n_rT_capped:,} events, {100*n_rT_capped/len(rT):.3f}%)")
        print(f"    d  capped at {d_cap:.4f} ({n_d_capped:,} events, {100*n_d_capped/len(d):.3f}%)")

    # Pass 1: determine needed features
    need = {'rt': {0}, 'lnrt': {0}, 'dphi': {0}, 'lndphi': {0}}
    composite_rt = set()
    composite_dphi = set()

    for o1, o2, _, _ in moments:
        b1, k1 = parse_moment(o1)
        b2, k2 = parse_moment(o2)
        if b1 is None or b2 is None:
            continue

        both_rt = _is_rt_type(b1) and _is_rt_type(b2)
        both_dphi = _is_dphi_type(b1) and _is_dphi_type(b2)

        if both_rt:
            if k1 > 0 and k2 > 0 and k1 <= max_k_rt and k2 <= max_k_rt:
                composite_rt.add((b1, k1, b2, k2))
            for base, k in [(b1, k1), (b2, k2)]:
                if k <= max_k_rt:
                    need[base].add(k)
        elif both_dphi:
            if k1 > 0 and k2 > 0 and k1 <= max_k_dphi and k2 <= max_k_dphi:
                composite_dphi.add((b1, k1, b2, k2))
            for base, k in [(b1, k1), (b2, k2)]:
                if k <= max_k_dphi:
                    need[base].add(k)
        else:
            for base, k in [(b1, k1), (b2, k2)]:
                max_k = max_k_rt if _is_rt_type(base) else max_k_dphi
                if k <= max_k:
                    need[base].add(k)

    def _basis(values, base, k, pow_cells_rT=None, pow_cells_lnrT=None,
               pow_cells_d=None, pow_cells_lnd=None):
        """Helper: returns the per-event or per-cell-exact array."""
        if k == 0:
            return np.ones(len(values), dtype=np.float64)
        if binned:
            if base == 'rt':
                return pow_cells_rT[k] if pow_cells_rT else rT_pow_cell[k]
            if base == 'lnrt':
                return pow_cells_lnrT[k] if pow_cells_lnrT else lnrT_pow_cell[k]
            if base == 'dphi':
                return pow_cells_d[k] if pow_cells_d else d_pow_cell[k]
            if base == 'lndphi':
                return pow_cells_lnd[k] if pow_cells_lnd else lnd_pow_cell[k]
        # event-level fallback
        if base in ('rt', 'dphi'):
            return values ** k
        return np.log(np.maximum(values, 1e-30)) ** k

    # Build F (rT features)
    F_cols, F_names = [np.ones(len(rT))], [('const', 0)]
    for k in sorted(need['rt'] - {0}):
        F_cols.append(_basis(rT, 'rt', k))
        F_names.append(('rt', k))
    for k in sorted(need['lnrt'] - {0}):
        F_cols.append(_basis(rT, 'lnrt', k))
        F_names.append(('lnrt', k))
    for b1, k1, b2, k2 in sorted(composite_rt):
        # Composite within rT: ⟨f1*f2⟩_cell ≠ ⟨f1⟩⟨f2⟩ in general; use cell-mean
        # product as small-bin approximation (exact in event mode).
        col = _basis(rT, b1, k1) * _basis(rT, b2, k2)
        name = _composite_key(b1, k1, b2, k2)
        F_cols.append(col)
        F_names.append((name, 0))
        print(f"    Composite F feature: {name}")

    # Build G (dphi features)
    G_cols, G_names = [np.ones(len(d))], [('const', 0)]
    for k in sorted(need['dphi'] - {0}):
        G_cols.append(_basis(d, 'dphi', k))
        G_names.append(('dphi', k))
    for k in sorted(need['lndphi'] - {0}):
        G_cols.append(_basis(d, 'lndphi', k))
        G_names.append(('lndphi', k))
    for b1, k1, b2, k2 in sorted(composite_dphi):
        col = _basis(d, b1, k1) * _basis(d, b2, k2)
        name = _composite_key(b1, k1, b2, k2)
        G_cols.append(col)
        G_names.append((name, 0))
        print(f"    Composite G feature: {name}")

    F = np.column_stack(F_cols)
    G = np.column_stack(G_cols)

    print(f"  F: {F.shape} features ({len(composite_rt)} composite)")
    print(f"  G: {G.shape} features ({len(composite_dphi)} composite)")

    # Report feature ranges after winsorization
    for i, name in enumerate(F_names):
        if name[0] != 'const':
            col = F[:, i]
            print(f"    F[{name}]: range=[{col.min():.4f}, {col.max():.4f}], std={col.std():.4f}")
    for j, name in enumerate(G_names):
        if name[0] != 'const':
            col = G[:, j]
            print(f"    G[{name}]: range=[{col.min():.4f}, {col.max():.4f}], std={col.std():.4f}")

    return F, G, F_names, G_names


def extract_pairs(F_names, G_names, moments, max_k_rt, max_k_dphi, sigma_scale=None):
    """Map moments to (i,j) feature pairs.

    sigma_scale: optional {(o1,o2): σ_scale_norm} (from compute_sigma_theory). When
    given, the returned per-pair σ is the PHYSICAL theory uncertainty
    σ = √(σ_scale² + σ_stat²) instead of the bare statistical σ."""
    print(f"\n[Extracting Moment Constraints]")

    F_idx = {name: i for i, name in enumerate(F_names)}
    G_idx = {name: j for j, name in enumerate(G_names)}

    moment_dict = {}
    n_skipped = 0
    skipped_names = []

    for o1, o2, val, unc in moments:
        b1, k1 = parse_moment(o1)
        b2, k2 = parse_moment(o2)

        if b1 is None or b2 is None:
            continue
        if k1 == 0 and k2 == 0:
            continue

        both_rt = _is_rt_type(b1) and _is_rt_type(b2)
        both_dphi = _is_dphi_type(b1) and _is_dphi_type(b2)
        b1_rt = _is_rt_type(b1)
        b2_rt = _is_rt_type(b2)

        fi, gj = None, None

        if both_rt:
            if k1 == 0:
                key_f = (b2, k2)
            elif k2 == 0:
                key_f = (b1, k1)
            else:
                key_f = (_composite_key(b1, k1, b2, k2), 0)
            key_g = ('const', 0)
            fi = F_idx.get(key_f)
            gj = G_idx.get(key_g)

        elif both_dphi:
            key_f = ('const', 0)
            if k1 == 0:
                key_g = (b2, k2)
            elif k2 == 0:
                key_g = (b1, k1)
            else:
                key_g = (_composite_key(b1, k1, b2, k2), 0)
            fi = F_idx.get(key_f)
            gj = G_idx.get(key_g)

        elif b1_rt and not b2_rt:
            key_f = (b1, k1) if k1 > 0 else ('const', 0)
            key_g = (b2, k2) if k2 > 0 else ('const', 0)
            fi = F_idx.get(key_f)
            gj = G_idx.get(key_g)

        elif not b1_rt and b2_rt:
            key_f = (b2, k2) if k2 > 0 else ('const', 0)
            key_g = (b1, k1) if k1 > 0 else ('const', 0)
            fi = F_idx.get(key_f)
            gj = G_idx.get(key_g)

        if fi is not None and gj is not None:
            ij = (fi, gj)
            if ij not in moment_dict:
                moment_dict[ij] = {'vals': [], 'uncs': [], 'scales': []}
            moment_dict[ij]['vals'].append(val)
            if unc is not None:
                moment_dict[ij]['uncs'].append(unc)
            if sigma_scale is not None:
                moment_dict[ij]['scales'].append(float(sigma_scale.get((str(o1), str(o2)), 0.0)))
        else:
            n_skipped += 1
            name = f"{b1}^{k1}×{b2}^{k2}"
            if name not in skipped_names:
                skipped_names.append(name)

    if n_skipped > 0:
        print(f"  WARNING: {n_skipped} moments skipped (no feature match):")
        for name in skipped_names[:10]:
            print(f"    - {name}")

    # Average duplicates
    pairs = []
    targets = []
    sigmas = []

    for (i, j), data in moment_dict.items():
        avg_val = np.mean(data['vals'])

        if data['uncs']:
            if len(data['uncs']) > 1:
                avg_unc = np.sqrt(np.mean([u**2 for u in data['uncs']]))
            else:
                avg_unc = data['uncs'][0]
        else:
            avg_unc = None

        pairs.append((i, j))
        targets.append(avg_val)
        if sigma_scale is not None:
            sc = np.sqrt(np.mean([s**2 for s in data['scales']])) if data.get('scales') else 0.0
            stat = avg_unc if avg_unc is not None else 0.0
            sigmas.append(float(np.sqrt(sc**2 + stat**2)))
        else:
            sigmas.append(avg_unc)

    print(f"  Unique constraints: {len(pairs)}")
    n_with_unc = sum(1 for s in sigmas if s is not None)
    print(f"  Constraints with target uncertainty: {n_with_unc}/{len(pairs)}")

    return (np.array(pairs, dtype=np.int64),
            np.array(targets, dtype=np.float64),
            sigmas)


# ========================================
# MaxEnt Model — Penalized Dual with Newton
# ========================================
class MaxEntDual:
    """
    Penalized dual MaxEnt:  L(λ) = log Z(λ) − Σ λ_k μ_k + ½ Σ σ_k² λ_k²

    Gradient:   ⟨g_k⟩_w − μ_k + σ_k² λ_k
    Hessian:    Cov_w(g_j, g_k) + diag(σ²)   [always PSD → convex]

    Newton step: Δλ = −H⁻¹ ∇L   (27×27 solve, trivial)
    """

    def __init__(self, F, G, pairs, targets, w0, sigmas_target=None,
                 F_names=None, G_names=None, cov_target=None):
        self.F = F                    # (N, n_F) float64
        self.G = G                    # (N, n_G) float64
        self.pairs = pairs            # (K, 2) int64 — constraint feature indices
        self.targets = targets        # (K,) float64 — target moment values
        # Signed-weight support: keep sign separately, work with log|w0|.
        # (Sherpa NLO/merged MC has ~12% negative-weight events; the old
        #  np.maximum(w0,1e-300) silently DROPPED them, biasing the tail.)
        self.sign0 = np.where(w0 >= 0.0, 1.0, -1.0)
        self.logw0 = np.log(np.maximum(np.abs(w0), 1e-300))
        self.F_names = F_names
        self.G_names = G_names
        self.K = len(pairs)           # number of constraints
        self.N = F.shape[0]

        # σ² for penalty term
        self._setup_sigmas(sigmas_target)
        # optional MATRIX penalty  ½ λᵀ Σ λ  (correlated systematics, e.g. scale-scheme
        # covariance + diag stat²). Overrides the diagonal reg_coef when provided.
        self.reg_mat = None
        if cov_target is not None:
            self.reg_mat = np.asarray(cov_target, dtype=np.float64)
            assert self.reg_mat.shape == (self.K, self.K)
            self.sigma = np.sqrt(np.maximum(np.diag(self.reg_mat), 1e-30))  # for pull printing

        # Lazy: m_prior computed on first access (saves a full pass over events
        # per candidate; selection makes ~100s of MaxEntDual instances per step)
        self.lam = np.zeros(self.K, dtype=np.float64)
        self.m_prior = None  # populated by _ensure_prior_moments() on demand

    def _ensure_prior_moments(self):
        if self.m_prior is None:
            _, self.m_prior, _ = self._compute_logZ_moments_cov(
                np.zeros(self.K, dtype=np.float64), need_cov=False)
        return self.m_prior

    def _setup_sigmas(self, sigmas_target):
        """Set up σ for the penalty term ½σ²λ²"""
        if sigmas_target is not None:
            sigma_list = []
            n_fallback = 0
            for sig in sigmas_target:
                if sig is not None and sig > 0:
                    sigma_list.append(sig)
                else:
                    # Fallback: use 10% of |target| as uncertainty
                    sigma_list.append(0.1 * abs(self.targets[len(sigma_list)]) + 1e-12)
                    n_fallback += 1
            self.sigma = np.array(sigma_list, dtype=np.float64)
            print(f"  Using target uncertainties for σ ({n_fallback} fallbacks)")
        else:
            # Default: 1% relative uncertainty → tight matching
            self.sigma = 0.01 * np.abs(self.targets) + 1e-12
            print(f"  No target uncertainties; using 1% relative as default σ")

        self.sigma2 = self.sigma ** 2
        # Regularizer coefficient: σ² (Bayesian default) or σ (linear, --reg_linear).
        # Linear regularizes high-σ (tail) moments less harshly.
        self.reg_coef = self.sigma if _REG_LINEAR else self.sigma2
        print(f"  σ range: [{self.sigma.min():.3e}, {self.sigma.max():.3e}]"
              f"{'  [linear reg]' if _REG_LINEAR else ''}")

    def _compute_logZ_moments_cov(self, lam, need_cov=True):
        """
        Single pass over events computing logZ, moments, and optionally covariance.

        Returns: (logZ, moments, cov_matrix)
            moments: (K,) array
            cov_matrix: (K, K) array or None
        """
        batch = 500_000
        N = self.N
        K = self.K
        i_idx = self.pairs[:, 0]
        j_idx = self.pairs[:, 1]

        # Pass 1: find global max of logits for numerical stability
        global_max = -np.inf
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]  # (batch, K)
            logits = self.logw0[a:b] + T @ lam
            m = logits.max()
            if m > global_max:
                global_max = m

        # Pass 2: accumulate Z, moments, and (optionally) covariance
        sum_exp = 0.0
        S1 = np.zeros(K, dtype=np.float64)
        if need_cov:
            S2 = np.zeros((K, K), dtype=np.float64)

        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            logits = self.logw0[a:b] + T @ lam
            w = self.sign0[a:b] * np.exp(logits - global_max)  # signed reweighted weight

            sum_w = w.sum()
            sum_exp += sum_w
            S1 += w @ T                       # (K,)

            if need_cov:
                wT = w[:, None] * T            # (batch, K) weighted
                S2 += wT.T @ T                 # (K, K)

        # Signed measure: a Newton trial step can drive the NET measure sum_exp
        # non-positive. Signal infeasibility (logZ=+inf) so the line search
        # rejects the step cleanly instead of producing nan from log(<=0).
        if not (sum_exp > 0) or not np.isfinite(sum_exp):
            return np.inf, np.full(K, np.nan), (np.full((K, K), np.nan) if need_cov else None)
        logZ = global_max + np.log(sum_exp)
        moments = S1 / sum_exp
        if need_cov:
            cov = S2 / sum_exp - np.outer(moments, moments)
        else:
            cov = None

        return logZ, moments, cov

    def dual_loss(self, lam):
        """L(λ) = log Z(λ) − λ·μ + ½ λᵀΣλ (matrix) or ½ Σ c_k λ_k² (diagonal)"""
        logZ, _, _ = self._compute_logZ_moments_cov(lam, need_cov=False)
        if self.reg_mat is not None:
            return logZ - lam @ self.targets + 0.5 * (lam @ (self.reg_mat @ lam))
        return logZ - lam @ self.targets + 0.5 * (self.reg_coef * lam * lam).sum()

    def dual_grad(self, lam):
        """∇L = ⟨g⟩_w − μ + Σλ (matrix) or cλ (diagonal)"""
        _, moments, _ = self._compute_logZ_moments_cov(lam, need_cov=False)
        reg = (self.reg_mat @ lam) if self.reg_mat is not None else self.reg_coef * lam
        return moments - self.targets + reg

    def dual_loss_grad_hess(self, lam):
        """Compute loss, gradient, and Hessian in a single pass over events."""
        logZ, moments, cov = self._compute_logZ_moments_cov(lam, need_cov=True)

        if self.reg_mat is not None:
            loss = logZ - lam @ self.targets + 0.5 * (lam @ (self.reg_mat @ lam))
            grad = moments - self.targets + self.reg_mat @ lam
            hess = cov + self.reg_mat
        else:
            loss = logZ - lam @ self.targets + 0.5 * (self.reg_coef * lam * lam).sum()
            grad = moments - self.targets + self.reg_coef * lam
            hess = cov + np.diag(self.reg_coef)

        return loss, grad, hess, moments

    def get_weights(self, lam=None):
        """Compute normalized weights for given λ"""
        if lam is None:
            lam = self.lam

        batch = 500_000
        N = self.N
        i_idx = self.pairs[:, 0]
        j_idx = self.pairs[:, 1]

        # Pass 1: global max
        global_max = -np.inf
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            logits = self.logw0[a:b] + T @ lam
            m = logits.max()
            if m > global_max:
                global_max = m

        # Pass 2: logZ
        sum_exp = 0.0
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            logits = self.logw0[a:b] + T @ lam
            sum_exp += (self.sign0[a:b] * np.exp(logits - global_max)).sum()
        if not (sum_exp > 0) or not np.isfinite(sum_exp):
            return np.full(N, np.nan)   # net measure non-positive: caller must handle
        logZ = global_max + np.log(sum_exp)   # net measure (sum_exp>0 since net σ>0)

        # Pass 3: weights (signed: negative-weight events stay negative)
        w = np.zeros(N, dtype=np.float64)
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b, i_idx] * self.G[a:b, j_idx]
            logits = self.logw0[a:b] + T @ lam
            w[a:b] = self.sign0[a:b] * np.exp(logits - logZ)

        return w


# ========================================
# MaxEnt Model — PyTorch GPU backend (MPS or CUDA)
# ========================================
class MaxEntDualGPU:
    """GPU port of MaxEntDual using PyTorch.

    Drop-in replacement: same constructor signature and methods. Internally
    keeps F, G, logw0 as torch tensors on the chosen device. All Newton-step
    computations (logits, exp, weighted sums, covariance) run on GPU.

    Numerics: float64 on CUDA (full precision), float32 on MPS (Apple Silicon
    has limited fp64 support). For typical problems, fp32 + log-sum-exp is
    accurate enough; switch to fp64 by setting MaxEntDualGPU.dtype=torch.float64
    if running on CUDA.
    """
    dtype = None  # set in __init__

    def __init__(self, F, G, pairs, targets, w0, sigmas_target=None,
                 F_names=None, G_names=None, device=None):
        """Accept either numpy arrays (uploads to GPU) or torch tensors
        already on GPU (zero-copy reuse). Logw0 also reused if torch tensor."""
        import torch
        self._torch = torch
        if device is None:
            if isinstance(F, torch.Tensor):
                device = F.device.type
            elif torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        # Use fp32 on MPS (no fp64), fp64 on CUDA/CPU
        self.dtype = torch.float32 if device == 'mps' else torch.float64

        # F, G: reuse if already torch tensor, else upload
        if isinstance(F, torch.Tensor):
            self.F = F
            self.G = G
        else:
            self.F = torch.from_numpy(np.ascontiguousarray(F, dtype=np.float64)
                                      ).to(device=device, dtype=self.dtype)
            self.G = torch.from_numpy(np.ascontiguousarray(G, dtype=np.float64)
                                      ).to(device=device, dtype=self.dtype)
        self.pairs = pairs    # numpy int array, kept on host
        self.i_idx = torch.from_numpy(pairs[:, 0].astype(np.int64)).to(device=device)
        self.j_idx = torch.from_numpy(pairs[:, 1].astype(np.int64)).to(device=device)
        self.targets = targets    # numpy float
        self.targets_t = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float64)).to(device=device, dtype=self.dtype)
        # logw0: reuse if torch tensor, else compute and upload
        if isinstance(w0, torch.Tensor):
            self.logw0 = w0
        else:
            logw0_np = np.log(np.maximum(w0, 1e-300))
            self.logw0 = torch.from_numpy(logw0_np).to(device=device, dtype=self.dtype)
        self.F_names = F_names
        self.G_names = G_names
        self.K = len(pairs)
        self.N = self.F.shape[0]

        self._setup_sigmas(sigmas_target)
        self.sigma2_t = torch.from_numpy(self.sigma2).to(device=device, dtype=self.dtype)

        # Lazy: skip diagnostic prior-moment pass for speed in selection
        self.lam = np.zeros(self.K, dtype=np.float64)
        self.m_prior = None

    def _ensure_prior_moments(self):
        if self.m_prior is None:
            _, self.m_prior, _ = self._compute_logZ_moments_cov(
                np.zeros(self.K, dtype=np.float64), need_cov=False)
        return self.m_prior

    def _setup_sigmas(self, sigmas_target):
        if sigmas_target is not None:
            sigma_list = []
            for sig in sigmas_target:
                if sig is not None and sig > 0:
                    sigma_list.append(sig)
                else:
                    sigma_list.append(0.1 * abs(self.targets[len(sigma_list)]) + 1e-12)
            self.sigma = np.array(sigma_list, dtype=np.float64)
        else:
            self.sigma = 0.01 * np.abs(self.targets) + 1e-12
        self.sigma2 = self.sigma ** 2

    def _compute_logZ_moments_cov(self, lam, need_cov=True):
        torch = self._torch
        lam_t = torch.from_numpy(np.ascontiguousarray(lam, dtype=np.float64)).to(
            device=self.device, dtype=self.dtype)
        # Per-event T is potentially huge so we batch in chunks
        batch = 2_000_000
        K = self.K
        N = self.N
        # Pass 1: global max
        global_max = torch.tensor(-float('inf'), device=self.device, dtype=self.dtype)
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b][:, self.i_idx] * self.G[a:b][:, self.j_idx]
            logits = self.logw0[a:b] + (T @ lam_t)
            m = logits.max()
            if m > global_max:
                global_max = m
        # Pass 2: weighted sums
        sum_exp = torch.zeros((), device=self.device, dtype=self.dtype)
        S1 = torch.zeros(K, device=self.device, dtype=self.dtype)
        S2 = torch.zeros((K, K), device=self.device, dtype=self.dtype) if need_cov else None
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b][:, self.i_idx] * self.G[a:b][:, self.j_idx]
            logits = self.logw0[a:b] + (T @ lam_t)
            w = torch.exp(logits - global_max)
            sum_exp = sum_exp + w.sum()
            S1 = S1 + (w[:, None] * T).sum(dim=0)
            if need_cov:
                wT = w[:, None] * T
                S2 = S2 + wT.T @ T
        logZ = (global_max + torch.log(sum_exp)).item()
        moments_t = S1 / sum_exp
        moments = moments_t.cpu().numpy().astype(np.float64)
        if need_cov:
            cov_t = S2 / sum_exp - torch.outer(moments_t, moments_t)
            cov = cov_t.cpu().numpy().astype(np.float64)
        else:
            cov = None
        return logZ, moments, cov

    def dual_loss(self, lam):
        logZ, _, _ = self._compute_logZ_moments_cov(lam, need_cov=False)
        return logZ - lam @ self.targets + 0.5 * (self.sigma2 * lam * lam).sum()

    def dual_loss_grad_hess(self, lam):
        logZ, moments, cov = self._compute_logZ_moments_cov(lam, need_cov=True)
        loss = logZ - lam @ self.targets + 0.5 * (self.sigma2 * lam * lam).sum()
        grad = moments - self.targets + self.sigma2 * lam
        hess = cov + np.diag(self.sigma2)
        return loss, grad, hess, moments

    def get_weights(self, lam=None):
        torch = self._torch
        if lam is None:
            lam = self.lam
        lam_t = torch.from_numpy(np.ascontiguousarray(lam, dtype=np.float64)).to(
            device=self.device, dtype=self.dtype)
        batch = 2_000_000
        N = self.N
        global_max = torch.tensor(-float('inf'), device=self.device, dtype=self.dtype)
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b][:, self.i_idx] * self.G[a:b][:, self.j_idx]
            logits = self.logw0[a:b] + (T @ lam_t)
            m = logits.max()
            if m > global_max:
                global_max = m
        sum_exp = torch.zeros((), device=self.device, dtype=self.dtype)
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b][:, self.i_idx] * self.G[a:b][:, self.j_idx]
            logits = self.logw0[a:b] + (T @ lam_t)
            sum_exp = sum_exp + torch.exp(logits - global_max).sum()
        logZ = global_max + torch.log(sum_exp)
        w = torch.empty(N, device=self.device, dtype=self.dtype)
        for a in range(0, N, batch):
            b = min(a + batch, N)
            T = self.F[a:b][:, self.i_idx] * self.G[a:b][:, self.j_idx]
            logits = self.logw0[a:b] + (T @ lam_t)
            w[a:b] = torch.exp(logits - logZ)
        return w.cpu().numpy().astype(np.float64)


# ========================================
# Newton Optimizer
# ========================================
def optimize_newton(model, max_steps=50, tol=1e-8, verbose=True):
    """
    Newton's method on the convex penalized dual.

    Each step: Δλ = −H⁻¹ g  where H = Cov_w(g_j, g_k) + diag(σ²)
    Backtracking line search for safety.

    Returns final loss.
    """
    lam = model.lam.copy()
    K = len(lam)
    lm = 0.0   # adaptive Levenberg-Marquardt damping (0 = pure Newton)

    print(f"\n[Newton Optimization: {K} constraints, tol={tol}]")

    for step in range(1, max_steps + 1):
        loss, grad, hess, moments = model.dual_loss_grad_hess(lam)

        grad_norm = np.max(np.abs(grad))
        resid = moments - model.targets
        pulls = resid / np.maximum(model.sigma, 1e-30)
        rms_pull = np.sqrt(np.mean(pulls**2))
        max_pull = np.max(np.abs(pulls))

        if verbose or step <= 3 or step % 5 == 0:
            print(f"  Step {step:3d}: Loss={loss:12.6f}  |∇|_∞={grad_norm:.3e}  "
                  f"RMS pull={rms_pull:.4f}  max|pull|={max_pull:.4f}")

        if grad_norm < tol:
            print(f"  Converged at step {step}: |∇|_∞ = {grad_norm:.3e} < {tol}")
            break

        # Adaptive Levenberg-Marquardt direction: solve (H + lm·diag(H)) Δλ = -g, raising the
        # damping lm only when a step fails to reduce the loss. On well-conditioned Hessians
        # lm stays ~0 (pure Newton, α=1 — identical to before); on collinear/ill-conditioned
        # sets lm grows so the direction is always usable, avoiding the 30× line-search
        # backtracking that made cold fits thrash. Same convex optimum, faster & robust path.
        diagH = np.maximum(np.diag(hess), 1e-30)
        c1 = 1e-4
        accepted = False
        for _try in range(12):
            try:
                dlam = np.linalg.solve(hess + (lm + 1e-12) * np.diag(diagH), -grad)
            except np.linalg.LinAlgError:
                lm = max(lm * 10.0, 1e-6); continue
            slope = grad @ dlam
            if slope > 0:                          # not a descent direction -> damp more
                lm = max(lm * 10.0, 1e-6); continue
            alpha = 1.0; lo = np.inf
            for _ in range(12):                    # short line search (good direction -> few backtracks)
                lam_trial = lam + alpha * dlam; lo = model.dual_loss(lam_trial)
                if np.isfinite(lo) and lo <= loss + c1 * alpha * slope: break
                alpha *= 0.5
            if np.isfinite(lo) and lo < loss:      # accept; relax damping for the next step
                lam = lam_trial; lm = lm * 0.3 if lm > 1e-12 else 0.0; accepted = True; break
            lm = max(lm * 10.0, 1e-6)              # step didn't help -> damp more, retry
        if not accepted:
            if verbose: print(f"  LM: no further progress at step {step}")
            break

        if verbose and (lm > 0 or alpha < 1.0):
            print(f"    (LM λ={lm:.1e}, α={alpha:.4f})")

    # Store result
    model.lam = lam

    # Final evaluation
    loss, grad, _, moments = model.dual_loss_grad_hess(lam)
    resid = moments - model.targets
    pulls = resid / np.maximum(model.sigma, 1e-30)
    chi2_equiv = np.sum(pulls**2)

    print(f"\n  Final: Loss={loss:.6f}, χ²_equiv={chi2_equiv:.4f}, "
          f"RMS pull={np.sqrt(np.mean(pulls**2)):.4f}, max|pull|={np.max(np.abs(pulls)):.4f}")
    print(f"  |∇|_∞ = {np.max(np.abs(grad)):.3e}")

    return loss


# ========================================
# Greedy Moment Selection by Triangle Divergence
# ========================================
def compute_td(prior_data, w, target_dists, n_events):
    """
    Total triangle divergence (rT + dphi) for given weights.
    Weights should already be for the first n_events events.
    """
    td = 0.0
    for dist_name, var_name in [('rTDist', 'rT'), ('dphiDist', 'd')]:
        if dist_name not in target_dists:
            continue
        edges = target_dists[dist_name]['edges']
        t_cen = target_dists[dist_name]['central']

        x = prior_data[var_name][:n_events]
        mask = (x >= edges[0]) & (x < edges[-1]) & np.isfinite(x)
        counts, _ = np.histogram(x[mask], bins=edges, weights=w[mask])
        widths = np.diff(edges)
        dens = counts / np.maximum(widths, 1e-300)

        # Zero out bins where target is zero
        dens = np.where(t_cen > 0, dens, 0.0)

        # Match area to target
        A_ref = np.sum(t_cen * widths)
        A = np.sum(dens * widths)
        if A > 0:
            dens *= A_ref / A

        td += triangle_divergence(dens, t_cen)

    return td


# Module-level flags set from args; checked in compute_td_split
_TD_LOG_DENSITY = False
_TARGET_REL_ACC = 0.0  # 0 = use original target unc; >0 = override with rel_acc*t_cen
_REG_LINEAR = False  # False = σ² regularizer (Bayesian); True = σ (linear)
_TD_STAT_AWARE = False  # True = include per-bin MC stat error in the χ² denominator
                        # (σ_eff² = σ_MC² + σ_theory²); makes the metric not chase
                        # statistical noise in low-count (tail) bins.
_TARGET_REL_FLOOR = 0.0  # 0 = no floor; >0 → σ = max(REL·t_cen, REL_FLOOR·max(t_cen))
                         # prevents χ² explosion in bins where theory ≈ 0 (e.g. dphi→π)


def compute_td_split(prior_data, w, target_dists, n_events):
    """Returns (td_rT, td_dphi) and (chi2pb_rT, chi2pb_dphi) for given weights.

    chi2pb = chi2/bin using target stat uncertainty (0.0 if no uncertainty available).
    If _TD_LOG_DENSITY is True, TD for BOTH rTDist and dphiDist is computed on
    log(density) — a relative (per-decade) metric so peak and tail count equally,
    and so the rT and dphi components share comparable units (the 'total' phase
    then balances them instead of being dominated by the larger plain-TD term).
    """
    tds = {}
    chi2pbs = {}
    for dist_name, var_name in [('rTDist', 'rT'), ('dphiDist', 'd')]:
        if dist_name not in target_dists:
            tds[dist_name] = 0.0
            chi2pbs[dist_name] = 0.0
            continue
        edges = target_dists[dist_name]['edges']
        t_cen = target_dists[dist_name]['central']
        t_unc = target_dists[dist_name].get('central_unc', None)

        x = prior_data[var_name][:n_events]
        mask = (x >= edges[0]) & (x < edges[-1]) & np.isfinite(x)
        counts, _ = np.histogram(x[mask], bins=edges, weights=w[mask])
        widths = np.diff(edges)
        dens = counts / np.maximum(widths, 1e-300)
        dens = np.where(t_cen > 0, dens, 0.0)

        A_ref = np.sum(t_cen * widths)
        A = np.sum(dens * widths)
        if A > 0:
            dens *= A_ref / A

        if _TD_LOG_DENSITY:
            # L1 distance on log-densities: Σ |log p - log q| × width
            # (triangle_divergence formula breaks for log values since
            #  (log p + log q) in the denominator can be negative)
            safe = (dens > 0) & (t_cen > 0)
            if np.any(safe):
                log_p = np.log(dens[safe])
                log_q = np.log(t_cen[safe])
                ws = widths[safe]
                tds[dist_name] = float(np.sum(np.abs(log_p - log_q) * ws))
            else:
                tds[dist_name] = 0.0
        else:
            tds[dist_name] = triangle_divergence(dens, t_cen)

        # Override target σ if --target_rel_acc is set (drives chi² ~ Σ(Δ/q)²)
        if _TARGET_REL_ACC > 0:
            t_unc_eff = _TARGET_REL_ACC * t_cen
            # Floor against bins where t_cen ≈ 0 (e.g. dphi→π) — relative error is
            # meaningless there. σ floored at REL_FLOOR × max(t_cen) treats those
            # bins on an absolute scale instead.
            if _TARGET_REL_FLOOR > 0:
                t_unc_eff = np.maximum(t_unc_eff, _TARGET_REL_FLOOR * float(np.max(t_cen)))
        else:
            t_unc_eff = t_unc
        # Error-aware: fold in the per-bin MC statistical error of the (area-
        # matched) reweighted density, σ_MC = sqrt(Σ w²)/width · (A_ref/A), so the
        # χ² doesn't demand accuracy beyond what the MC statistics support.
        if _TD_STAT_AWARE:
            # FIXED Poisson error from the UNWEIGHTED event count per bin:
            # σ_MC = t_cen / sqrt(N_bin). Independent of weights, so it can't be
            # gamed by reweighting that spreads weights / lowers N_eff.
            cnt, _ = np.histogram(x[mask], bins=edges)
            sig_mc = np.where((t_cen > 0) & (cnt > 0),
                              t_cen / np.sqrt(np.maximum(cnt, 1)), 0.0)
            if t_unc_eff is None:
                t_unc_eff = sig_mc
            else:
                t_unc_eff = np.sqrt(np.asarray(t_unc_eff) ** 2 + sig_mc ** 2)
        if t_unc_eff is not None:
            chi2pb, _ = chi2_per_bin(dens, t_cen, t_unc_eff)
        else:
            chi2pb = 0.0
        chi2pbs[dist_name] = chi2pb

    return (tds.get('rTDist', 0.0), tds.get('dphiDist', 0.0),
            chi2pbs.get('rTDist', 0.0), chi2pbs.get('dphiDist', 0.0))


def _feature_power(name_tuple):
    """Compute the total power of a feature.
    ('const', 0) -> 0, ('rt', 3) -> 3, ('lnrt', 2) -> 2,
    ('lnrt^2*rt^1', 0) -> 3 (sum of all exponents in composite)
    """
    name, k = name_tuple
    if k > 0:
        return k  # simple feature: ('rt', k) or ('lnrt', k) etc.
    if name == 'const':
        return 0
    # Composite: parse 'lnrt^2*rt^1' -> sum of exponents
    import re
    powers = re.findall(r'\^(\d+)', name)
    return sum(int(p) for p in powers) if powers else 0


def _pair_total_power(pair, F_names, G_names):
    """Total power of a moment constraint = power(F feature) + power(G feature)."""
    i, j = pair
    return _feature_power(F_names[i]) + _feature_power(G_names[j])


# ── Multiprocessing support for candidate evaluation ──
_MP_DATA = {}  # Module-level shared data for worker processes

# ── Multiprocessing support for variation reweighting ──
_MP_VAR_DATA = {}  # Module-level shared data for variation workers


def _mp_var_init(F, G, w0, F_names, G_names, hist_edges_data,
                 prior_rT, prior_d, prior_pT, prior_m,
                 max_newton_steps, newton_tol):
    """Initialize variation worker with shared arrays."""
    _MP_VAR_DATA['F'] = F
    _MP_VAR_DATA['G'] = G
    _MP_VAR_DATA['w0'] = w0
    _MP_VAR_DATA['F_names'] = F_names
    _MP_VAR_DATA['G_names'] = G_names
    _MP_VAR_DATA['hist_edges_data'] = hist_edges_data
    _MP_VAR_DATA['prior_rT'] = prior_rT
    _MP_VAR_DATA['prior_d'] = prior_d
    _MP_VAR_DATA['prior_pT'] = prior_pT
    _MP_VAR_DATA['prior_m'] = prior_m
    _MP_VAR_DATA['max_newton_steps'] = max_newton_steps
    _MP_VAR_DATA['newton_tol'] = newton_tol


def _mp_reweight_variation(task):
    """Reweight a single scale variation in a worker process."""
    import io, contextlib

    scale_fo, scale_res, pairs_arr, targets_arr, sigmas_list, warm_lam, do_stat_band, sigmas_orig = task

    F = _MP_VAR_DATA['F']
    G = _MP_VAR_DATA['G']
    w0 = _MP_VAR_DATA['w0']
    F_names = _MP_VAR_DATA['F_names']
    G_names = _MP_VAR_DATA['G_names']
    hist_edges_data = _MP_VAR_DATA['hist_edges_data']
    max_steps = _MP_VAR_DATA['max_newton_steps']
    tol = _MP_VAR_DATA['newton_tol']

    # Reconstruct hist_edges with prior data references
    hist_edges = {}
    prior_rT = _MP_VAR_DATA['prior_rT']
    prior_d = _MP_VAR_DATA['prior_d']
    prior_pT = _MP_VAR_DATA['prior_pT']
    prior_m = _MP_VAR_DATA['prior_m']
    for key, edges in hist_edges_data.items():
        if key == 'rT':
            hist_edges[key] = (prior_rT, edges)
        elif key == 'dphi':
            hist_edges[key] = (prior_d, edges)
        elif key == 'pT':
            hist_edges[key] = (prior_pT, edges)
        elif key == 'mass':
            hist_edges[key] = (prior_m, edges)

    results = []

    # Main variation fit
    with contextlib.redirect_stdout(io.StringIO()):
        model_var = MaxEntDual(F, G, pairs_arr, targets_arr, w0,
                               sigmas_target=sigmas_list,
                               F_names=F_names, G_names=G_names)
        if warm_lam is not None and len(warm_lam) == len(model_var.lam):
            model_var.lam[:] = warm_lam.copy()
        optimize_newton(model_var, max_steps=max_steps,
                        tol=tol, verbose=False)

    w_rew_var = model_var.get_weights()
    n_eff_var = 100 * (np.sum(w_rew_var)**2 / np.sum(w_rew_var**2)) / len(w_rew_var)

    _, m_var, _ = model_var._compute_logZ_moments_cov(model_var.lam, need_cov=False)
    pulls_var = (m_var - targets_arr) / np.maximum(model_var.sigma, 1e-30)
    rms_var = float(np.sqrt(np.mean(pulls_var**2)))

    hists = precompute_variation_hists(w_rew_var, hist_edges)
    lam_var = model_var.lam.copy()

    results.append(((scale_fo, scale_res), hists, lam_var, rms_var, n_eff_var))

    del w_rew_var, model_var

    # Stat band fits
    if do_stat_band and sigmas_orig is not None:
        sigmas_orig_arr = np.array(sigmas_orig)
        for direction in ['stat_up', 'stat_down']:
            if direction == 'stat_up':
                tgt_shifted = targets_arr + sigmas_orig_arr
            else:
                tgt_shifted = targets_arr - sigmas_orig_arr

            with contextlib.redirect_stdout(io.StringIO()):
                m_sb = MaxEntDual(F, G, pairs_arr, tgt_shifted, w0,
                                  sigmas_target=sigmas_list,
                                  F_names=F_names, G_names=G_names)
                m_sb.lam = lam_var.copy()
                optimize_newton(m_sb, max_steps=max_steps,
                                tol=tol, verbose=False)

            w_sb = m_sb.get_weights()
            sb_hists = precompute_variation_hists(w_sb, hist_edges)
            sb_key = (scale_fo, scale_res, direction)
            results.append((sb_key, sb_hists, m_sb.lam.copy(), 0.0, 0.0))
            del w_sb, m_sb

    return results


def _mp_init_worker(F_sub, G_sub, w_sub, all_pairs, all_targets, all_sigmas,
                    F_names, G_names, prior_data, target_dists, n_ev,
                    use_gpu=False):
    """Initialize worker process with shared data (called once per worker via fork COW).
    If use_gpu, also pre-upload F, G, logw0 to GPU once and cache them.
    """
    _MP_DATA['F_sub'] = F_sub
    _MP_DATA['G_sub'] = G_sub
    _MP_DATA['w_sub'] = w_sub
    _MP_DATA['all_pairs'] = all_pairs
    _MP_DATA['all_targets'] = all_targets
    _MP_DATA['all_sigmas'] = all_sigmas
    _MP_DATA['F_names'] = F_names
    _MP_DATA['G_names'] = G_names
    _MP_DATA['prior_data'] = prior_data
    _MP_DATA['target_dists'] = target_dists
    _MP_DATA['n_ev'] = n_ev
    _MP_DATA['use_gpu'] = use_gpu
    if use_gpu:
        import torch
        device = ('cuda' if torch.cuda.is_available()
                  else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        dtype = torch.float32 if device == 'mps' else torch.float64
        _MP_DATA['_gpu_device'] = device
        _MP_DATA['_gpu_F'] = torch.from_numpy(np.ascontiguousarray(F_sub, dtype=np.float64)
                                              ).to(device=device, dtype=dtype)
        _MP_DATA['_gpu_G'] = torch.from_numpy(np.ascontiguousarray(G_sub, dtype=np.float64)
                                              ).to(device=device, dtype=dtype)
        logw0 = np.log(np.maximum(w_sub, 1e-300))
        _MP_DATA['_gpu_logw0'] = torch.from_numpy(logw0).to(device=device, dtype=dtype)


def _mp_eval_candidate(args):
    """Evaluate a single candidate in a worker process. Must be module-level for pickling."""
    idx, selected_, best_lam_, max_steps, tol = args

    use_gpu = _MP_DATA.get('use_gpu', False)
    if use_gpu:
        F_sub = _MP_DATA['_gpu_F']
        G_sub = _MP_DATA['_gpu_G']
        w_sub = _MP_DATA['_gpu_logw0']  # already log + on-device
    else:
        F_sub = _MP_DATA['F_sub']
        G_sub = _MP_DATA['G_sub']
        w_sub = _MP_DATA['w_sub']
    all_pairs = _MP_DATA['all_pairs']
    all_targets = _MP_DATA['all_targets']
    all_sigmas = _MP_DATA['all_sigmas']
    F_names = _MP_DATA['F_names']
    G_names = _MP_DATA['G_names']
    prior_data = _MP_DATA['prior_data']
    target_dists = _MP_DATA['target_dists']
    n_ev = _MP_DATA['n_ev']

    import io, contextlib

    trial = selected_ + [idx]
    pairs_t = all_pairs[trial]
    targets_t = all_targets[trial]
    sigmas_t = [all_sigmas[i_] for i_ in trial]

    Cls = MaxEntDualGPU if use_gpu else MaxEntDual
    with contextlib.redirect_stdout(io.StringIO()):
        model = Cls(F_sub, G_sub, pairs_t, targets_t, w_sub,
                    sigmas_target=sigmas_t,
                    F_names=F_names, G_names=G_names)
        if len(selected_) > 0:
            model.lam[:len(selected_)] = best_lam_.copy()
        optimize_newton(model, max_steps=max_steps,
                        tol=tol, verbose=False)

    dual = model.dual_loss(model.lam)

    w_rew = model.get_weights()
    w_rew = w_rew / w_rew.sum()
    td_rT, td_d, chi2pb_rT, chi2pb_d = compute_td_split(prior_data, w_rew, target_dists, n_ev)
    td = td_rT + td_d
    neff = (np.sum(w_rew)**2 / np.sum(w_rew**2)) / n_ev
    return (dual, td, td_rT, td_d, chi2pb_rT, chi2pb_d, idx, model.lam.copy(), neff)


def greedy_select_by_td(F, G, all_pairs, all_targets, all_sigmas, w0,
                        prior_data, target_dists, F_names, G_names,
                        max_moments=30, n_events=1_000_000,
                        newton_steps=30, newton_tol=1e-7,
                        screen_top_k=15, screen_steps=2,
                        n_workers=1, min_improvement_pct=0.5,
                        min_steps=5, max_component_worsen=None,
                        preselected_indices=None, no_backward=False,
                        select_strategy='total', use_gpu=False,
                        phase1_pure_rt=False):
    """
    Greedy forward selection minimizing triangle divergence (TD).

    Two-stage per step:
      1. SCREEN: 2 Newton steps on all candidates -> rough TD ranking
      2. REFINE: full Newton on top-K candidates -> pick winner by TD

    Dual loss (KL) tracked as diagnostic.

    Returns: list of selected indices into all_pairs, and the selection log.
    """
    import io, contextlib

    n_ev = min(n_events, F.shape[0])
    F_sub = F[:n_ev]
    G_sub = G[:n_ev]
    w_sub = w0[:n_ev]

    n_candidates = len(all_pairs)
    available = list(range(n_candidates))
    selected = []
    _forced_set = set()

    # Prior TD baseline
    w_prior_norm = w_sub / w_sub.sum()
    td_rT_0, td_d_0, chi2pb_rT_0, chi2pb_d_0 = compute_td_split(
        prior_data, w_prior_norm, target_dists, n_ev)
    td_baseline = td_rT_0 + td_d_0

    # Baseline dual at lambda=0
    dual_baseline = np.log(w_sub.sum())

    # Total power for each candidate (display only)
    candidate_powers = {}
    for idx in available:
        candidate_powers[idx] = _pair_total_power(all_pairs[idx], F_names, G_names)

    print(f"\n{'='*80}")
    print(f"[Greedy Moment Selection by Triangle Divergence]")
    print(f"  Candidates: {n_candidates}")
    print(f"  Events: {n_ev:,}")
    print(f"  Screening: {screen_steps} Newton steps -> top {screen_top_k} -> full solve")
    print(f"  Workers: {n_workers}")
    print(f"  Min steps: {min_steps}")
    print(f"  Baseline TD:     {td_baseline:.4f}  (rT={td_rT_0:.4f}, dphi={td_d_0:.4f})")
    print(f"  Baseline χ²/bin: rT={chi2pb_rT_0:.2f}, dphi={chi2pb_d_0:.2f}")
    if chi2pb_rT_0 > 50 or chi2pb_d_0 > 50:
        print(f"  WARNING: large prior χ²/bin — prior is far from this theory target.")
        print(f"           Large corrections needed; component TD tension likely.")
    print(f"  Baseline dual: {dual_baseline:.6f}")
    _cw_str = f"{100*max_component_worsen:.0f}% above running best" if max_component_worsen is not None else "disabled"
    print(f"  Max component worsen: {_cw_str}")
    print(f"  Selection strategy: {select_strategy}")
    print(f"{'='*80}")

    best_lam = np.array([], dtype=np.float64)
    selection_log = []
    dual_current = dual_baseline
    td_current = td_baseline
    td_rT_current = td_rT_0
    td_d_current = td_d_0
    metric_current = chi2pb_rT_0 if select_strategy == 'chi2' else td_baseline
    chi2_rT_locked = None  # set when leaving phase 1 of chi2_phased; ceiling for phase 2
    # Running best (minimum) per component — used for degradation threshold
    td_rT_best = td_rT_0
    td_d_best  = td_d_0

    # -- Pre-seed forced moments (never removed by backward step) --
    if preselected_indices:
        _forced_set = set(preselected_indices)
        selected = list(preselected_indices)
        available = [i for i in available if i not in _forced_set]
        print(f"\n  [Pre-seeding {len(selected)} forced moment(s)]")
        for fi in selected:
            i_, j_ = all_pairs[fi]
            print(f"    + {_display_name(F_names[i_], G_names[j_])}")
        pairs_pre = all_pairs[selected]
        targets_pre = all_targets[selected]
        sigmas_pre = [all_sigmas[k] for k in selected]
        model_pre = MaxEntDual(F_sub, G_sub, pairs_pre, targets_pre, w_sub,
                               sigmas_target=sigmas_pre,
                               F_names=F_names, G_names=G_names)
        optimize_newton(model_pre, max_steps=newton_steps, tol=newton_tol, verbose=False)
        best_lam = model_pre.lam.copy()
        dual_current = model_pre.dual_loss(model_pre.lam)
        w_pre = model_pre.get_weights()
        w_pre = w_pre / w_pre.sum()
        td_rT_current, td_d_current, _, _ = compute_td_split(
            prior_data, w_pre, target_dists, n_ev)
        td_current = td_rT_current + td_d_current
        td_rT_best = td_rT_current
        td_d_best  = td_d_current
        print(f"  Pre-seed TD: {td_current:.4f}  (rT={td_rT_current:.4f}, dphi={td_d_current:.4f})")

    def _eval_candidate(idx, selected_, best_lam_, max_steps, tol):
        """Evaluate candidate (serial fallback). Returns (dual, td, td_rT, td_d, idx, lam, neff)."""
        return _mp_eval_candidate((idx, selected_, best_lam_, max_steps, tol))

    # Initialize shared data (needed for both serial and parallel paths)
    _mp_init_worker(F_sub, G_sub, w_sub, all_pairs, all_targets,
                    all_sigmas, F_names, G_names, prior_data,
                    target_dists, n_ev, use_gpu=use_gpu)

    # Initialize multiprocessing pool (created once, reused across steps)
    _pool = None
    if n_workers > 1:
        import multiprocessing as mp
        import sys, platform
        # macOS Apple Silicon: fork is unsafe by default due to Obj-C runtime
        # but fine for pure NumPy workloads. Set env var to suppress crash.
        if platform.system() == 'Darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        ctx = mp.get_context('fork')
        _pool = ctx.Pool(processes=n_workers, initializer=_mp_init_worker,
                         initargs=(F_sub, G_sub, w_sub, all_pairs, all_targets,
                                   all_sigmas, F_names, G_names, prior_data,
                                   target_dists, n_ev, use_gpu))
        print(f"  Started pool with {n_workers} workers (fork)")

    def _run_batch(indices, selected_, best_lam_, max_steps, tol):
        """Run evaluations, serial or parallel."""
        if _pool is None or len(indices) < 4:
            results = []
            for count, idx in enumerate(indices):
                r = _eval_candidate(idx, selected_, best_lam_, max_steps, tol)
                results.append(r)
                if (count + 1) % 10 == 0 or count == len(indices) - 1:
                    print(f"    {count+1}/{len(indices)}...", end='\r')
            print()
            return results
        else:
            tasks = [(idx, selected_, best_lam_, max_steps, tol)
                     for idx in indices]
            results = []
            for count, r in enumerate(_pool.imap_unordered(_mp_eval_candidate, tasks)):
                results.append(r)
                if (count + 1) % 20 == 0 or count == len(indices) - 1:
                    print(f"    {count+1}/{len(indices)}...", end='\r')
            print()
            return results

    # -- Classify candidates by type (for phased strategies) --
    def _moment_category(idx):
        """Classify moment as 'rT', 'dphi', or 'cross'."""
        i, j = all_pairs[idx]
        f_const = (F_names[i] == ('const', 0))
        g_const = (G_names[j] == ('const', 0))
        if not f_const and g_const:
            return 'rT'
        elif f_const and not g_const:
            return 'dphi'
        else:
            return 'cross'

    candidate_categories = {idx: _moment_category(idx) for idx in range(n_candidates)}

    # Phase management for phased strategies
    if select_strategy == 'rT_then_dphi':
        phases = [
            ('rT',    lambda x: x[2], {'rT'}),           # minimize TD_rT, rT moments only
            ('dphi',  lambda x: x[3], {'dphi'}),          # minimize TD_dphi, dphi moments only
            ('cross', lambda x: x[1], {'cross', 'rT', 'dphi'}),  # minimize total TD, all remaining
        ]
        phase_idx = 0
        phase_name, phase_sort_key, phase_cats = phases[0]
        phase_td_prev = td_rT_0
        print(f"\n  Phase 1: rT marginals (minimize TD_rT)")
    elif select_strategy == 'td_phased':
        # 2-phase TD selection, all categories allowed in both phases:
        #   Phase 1: minimize TD_rT (with all candidates: rT, dphi, cross)
        #   Phase 2: minimize total TD (rT + dphi)
        phases = [
            ('td_rT',    lambda x: x[2], {'rT', 'dphi', 'cross'}),
            ('td_total', lambda x: x[1], {'rT', 'dphi', 'cross'}),
        ]
        phase_idx = 0
        phase_name, phase_sort_key, phase_cats = phases[0]
        phase_td_prev = td_rT_0
        print(f"\n  Phase 1: minimize TD_rT (all categories allowed)")
    elif select_strategy == 'chi2_phased':
        # 2-phase chi² selection:
        #   Phase 1: minimize chi2pb_rT using rT-dep moments
        #            (pure rT only if phase1_pure_rt=True, else rT+cross)
        #   Phase 2: minimize total chi² (rT + dphi) with ALL moments
        # Phase 2's metric naturally penalizes any moment that degrades chi²_rT.
        phase1_cats = {'rT'} if phase1_pure_rt else {'rT', 'cross'}
        phases = [
            ('chi2_rT',    lambda x: x[4],         phase1_cats),
            ('chi2_total', lambda x: x[4] + x[5],  {'rT', 'dphi', 'cross'}),
        ]
        phase_idx = 0
        phase_name, phase_sort_key, phase_cats = phases[0]
        phase_td_prev = chi2pb_rT_0
        print(f"\n  Phase 1: rT-dep moments (minimize chi2pb_rT)")
    else:
        phases = None

    for step in range(max_moments):
        if not available:
            print(f"\n  All candidates exhausted")
            break

        # Filter candidates for phased strategy
        if select_strategy in ('rT_then_dphi', 'chi2_phased', 'td_phased'):
            phase_available = [idx for idx in available
                               if candidate_categories[idx] in phase_cats]
            if not phase_available:
                # Advance to next phase
                if phase_idx + 1 < len(phases):
                    phase_idx += 1
                    phase_name, phase_sort_key, phase_cats = phases[phase_idx]
                    phase_td_prev = td_rT_current if phase_name == 'rT' else (
                        td_d_current if phase_name == 'dphi' else td_current)
                    print(f"\n  Phase {phase_idx+1}: {phase_name} "
                          f"({'minimize TD_'+phase_name if phase_name != 'cross' else 'minimize total TD'})")
                    phase_available = [idx for idx in available
                                       if candidate_categories[idx] in phase_cats]
                if not phase_available:
                    print(f"\n  No candidates left for any phase")
                    break
            step_available = phase_available
        elif select_strategy == 'chi2':
            # chi2_rT mode: always restrict to rT-dependent candidates
            # (rT marginal or rT×dphi cross terms; exclude dphi-only).
            step_available = [idx for idx in available
                              if candidate_categories[idx] in ('rT', 'cross')]
            if step == 0:
                print(f"  [chi2 mode: restricting to {len(step_available)} "
                      f"rT-dependent candidates of {len(available)}]")
        else:
            step_available = available

        n_avail = len(step_available)
        actual_top_k = min(screen_top_k, n_avail)

        # -- Stage 1: Quick screen --
        # Use the SAME ranking metric as the refine stage (chi2pb_rT for
        # chi2 modes, total TD otherwise) so the shortlist isn't pre-biased.
        if select_strategy in ('rT_then_dphi', 'chi2_phased', 'td_phased'):
            screen_key = phase_sort_key
        elif select_strategy == 'chi2':
            screen_key = lambda x: x[4]   # chi2pb_rT
        elif select_strategy == 'alternating':
            screen_key = (lambda x: x[2]) if (step + 1) % 2 == 1 else (lambda x: x[3])
        else:
            screen_key = lambda x: x[1]   # total TD

        if n_avail > actual_top_k:
            print(f"\n  --- Step {step+1}: screening {n_avail} candidates "
                  f"({screen_steps} Newton steps) ---")

            screen_results = _run_batch(
                step_available, selected, best_lam, screen_steps, newton_tol)

            screen_results.sort(key=screen_key)
            shortlist = [r[6] for r in screen_results[:actual_top_k]]

            print(f"    Top {actual_top_k} after screening (ranked by {focus if 'focus' in dir() else select_strategy}):")
            for rank, (dual_s, td_s, _, _, c2_rT, c2_d, idx_s, _, _) in enumerate(screen_results[:actual_top_k]):
                i_, j_ = all_pairs[idx_s]
                nm = _display_name(F_names[i_], G_names[j_])
                pw = candidate_powers[idx_s]
                print(f"      {rank+1:2d}. {nm:<28} TD={td_s:.1f}  χ²/bin: rT={c2_rT:.2f} d={c2_d:.2f}  (pw={pw})")
        else:
            shortlist = step_available[:]
            print(f"\n  --- Step {step+1}: testing {n_avail} candidates (full) ---")

        # -- Stage 2: Full Newton on shortlist --
        refine_results = _run_batch(
            shortlist, selected, best_lam, newton_steps, newton_tol)

        if not refine_results:        # candidate pool exhausted — stop gracefully
            print(f"\n  No candidates left to refine — stopping at {len(selected)} moments")
            break

        # Pick best by TD — strategy determines which component to minimize
        # x[1]=total TD, x[2]=TD_rT, x[3]=TD_dphi
        if select_strategy == 'alternating':
            if (step + 1) % 2 == 1:
                sort_key = lambda x: x[2]
                focus = 'rT'
            else:
                sort_key = lambda x: x[3]
                focus = 'dphi'
        elif select_strategy in ('rT_then_dphi', 'chi2_phased', 'td_phased'):
            sort_key = phase_sort_key
            focus = phase_name
        elif select_strategy == 'chi2':
            # rT-prioritized: minimize chi2pb_rT only (analog of rT_then_dphi for χ²).
            sort_key = lambda x: x[4]
            focus = 'chi2_rT'
        else:
            sort_key = lambda x: x[1]
            focus = 'total'

        if max_component_worsen is not None:
            rT_thresh = td_rT_best * (1.0 + max_component_worsen)
            d_thresh  = td_d_best  * (1.0 + max_component_worsen)
            constrained = [r for r in refine_results
                           if r[2] <= rT_thresh and r[3] <= d_thresh]
            if constrained:
                best_r = min(constrained, key=sort_key)
            else:
                best_r = min(refine_results, key=sort_key)
                print(f"  [Note] All candidates worsen a component beyond threshold "
                      f"(rT<={rT_thresh:.1f}, dphi<={d_thresh:.1f}); "
                      f"using unconstrained best.")
        else:
            best_r = min(refine_results, key=sort_key)
        (best_dual, best_td, best_td_rT, best_td_d,
         best_chi2pb_rT, best_chi2pb_d,
         best_idx, best_lam_new, best_neff) = best_r

        # Report
        i_, j_ = all_pairs[best_idx]
        name = _display_name(F_names[i_], G_names[j_])
        dual_drop = dual_current - best_dual
        td_drop = td_current - best_td
        td_pct_step = 100 * td_drop / td_current if td_current > 0 else 0
        td_pct_cum = 100 * (1 - best_td / td_baseline) if td_baseline > 0 else 0

        # Metric-aware step improvement (stopping uses this when select_strategy='chi2')
        if select_strategy == 'chi2':
            best_metric = best_chi2pb_rT
            metric_drop = metric_current - best_metric
            metric_pct_step = 100 * metric_drop / metric_current if metric_current > 0 else 0
        else:
            best_metric = best_td
            metric_drop = td_drop
            metric_pct_step = td_pct_step

        # Per-component degradation vs running best
        rT_vs_base  = 100 * (best_td_rT - td_rT_best) / td_rT_best if td_rT_best > 0 else 0
        d_vs_base   = 100 * (best_td_d  - td_d_best)  / td_d_best  if td_d_best  > 0 else 0
        if max_component_worsen is not None:
            rT_degraded = best_td_rT > td_rT_best * (1.0 + max_component_worsen)
            d_degraded  = best_td_d  > td_d_best  * (1.0 + max_component_worsen)
        else:
            rT_degraded = False
            d_degraded  = False

        # Compute pulls
        trial = selected + [best_idx]
        pairs_t = all_pairs[trial]
        targets_t = all_targets[trial]
        sigmas_t = [all_sigmas[k] for k in trial]
        with contextlib.redirect_stdout(io.StringIO()):
            model_check = MaxEntDual(F_sub, G_sub, pairs_t, targets_t, w_sub,
                                      sigmas_target=sigmas_t,
                                      F_names=F_names, G_names=G_names)
        model_check.lam = best_lam_new.copy()
        _, m_check, _ = model_check._compute_logZ_moments_cov(model_check.lam, need_cov=False)
        pulls_check = (m_check - targets_t) / np.maximum(model_check.sigma, 1e-30)
        rms_pull = np.sqrt(np.mean(pulls_check**2))

        pw = candidate_powers[best_idx]
        entry = {
            'step': step + 1,
            'name': name,
            'idx': best_idx,
            'power': pw,
            'dual': best_dual,
            'delta_dual': dual_drop,
            'td': best_td,
            'td_rT': best_td_rT,
            'td_dphi': best_td_d,
            'chi2pb_rT': best_chi2pb_rT,
            'chi2pb_dphi': best_chi2pb_d,
            'delta_td': td_drop,
            'pct_improvement': td_pct_cum,
            'rms_pull': rms_pull,
            'neff': best_neff,
        }
        selection_log.append(entry)

        # Component degradation annotation
        comp_note = ''
        if rT_degraded:
            comp_note += f'  !! rT TD {rT_vs_base:+.0f}% vs baseline'
        if d_degraded:
            comp_note += f'  !! dphi TD {d_vs_base:+.0f}% vs baseline'

        focus_tag = f'  [{focus}]' if select_strategy in ('alternating', 'rT_then_dphi') else ''
        print(f"  {step+1:2d}. +{name:<25}  "
              f"TD={best_td:.1f} (rT={best_td_rT:.1f} d={best_td_d:.1f}) [{td_pct_cum:+.1f}%]  "
              f"χ²/bin: rT={best_chi2pb_rT:.1f} d={best_chi2pb_d:.1f}  "
              f"N_eff={100*best_neff:.1f}%{comp_note}{focus_tag}")

        # Stopping logic
        in_grace = (step + 1 < min_steps)

        # Phase-specific tracking
        if select_strategy in ('rT_then_dphi', 'chi2_phased', 'td_phased'):
            if phase_name == 'rT':
                phase_td_new = best_td_rT
            elif phase_name == 'dphi':
                phase_td_new = best_td_d
            elif phase_name == 'chi2_rT':
                phase_td_new = best_chi2pb_rT
            elif phase_name == 'chi2_dphi':
                phase_td_new = best_chi2pb_d
            elif phase_name == 'chi2_total':
                phase_td_new = best_chi2pb_rT + best_chi2pb_d
            elif phase_name == 'td_rT':
                phase_td_new = best_td_rT
            elif phase_name == 'td_total':
                phase_td_new = best_td
            else:
                phase_td_new = best_td
            phase_td_drop = phase_td_prev - phase_td_new
            phase_pct_step = 100 * phase_td_drop / phase_td_prev if phase_td_prev > 0 else 0

        if in_grace:
            if td_drop < 0:
                print(f"     (TD worse, grace period: step {step+1} < min_steps={min_steps})")
            if rT_degraded or d_degraded:
                print(f"     (component degraded, grace period)")
        elif select_strategy in ('rT_then_dphi', 'chi2_phased', 'td_phased'):
            # Phase-specific stopping: saturated → advance to next phase
            advance_phase = False
            reject_moment = False
            if phase_td_drop < 0:
                print(f"  Phase '{phase_name}' TD got worse — advancing to next phase")
                selection_log.pop()
                reject_moment = True
                advance_phase = True
            elif phase_pct_step < min_improvement_pct and step > 0:
                print(f"  Phase '{phase_name}' saturated "
                      f"({phase_pct_step:.2f}% < {min_improvement_pct}%) — advancing")
                advance_phase = True

            if advance_phase:
                if not reject_moment:
                    selected.append(best_idx)
                    available.remove(best_idx)
                    dual_current = best_dual
                    td_current = best_td
                    td_rT_current = best_td_rT
                    td_d_current = best_td_d
                    best_lam = best_lam_new
                    td_rT_best = min(td_rT_best, best_td_rT)
                    td_d_best  = min(td_d_best,  best_td_d)
                if phase_idx + 1 < len(phases):
                    # Lock in chi²_rT achieved by phase 1 (used as ceiling in phase 2)
                    if phase_name == 'chi2_rT':
                        chi2_rT_locked = best_chi2pb_rT
                        print(f"  [Locking χ²_rT={chi2_rT_locked:.3f} as ceiling for phase 2]")
                    phase_idx += 1
                    phase_name, phase_sort_key, phase_cats = phases[phase_idx]
                    if phase_name == 'rT':
                        phase_td_prev = td_rT_current
                    elif phase_name == 'dphi':
                        phase_td_prev = td_d_current
                    elif phase_name == 'chi2_rT':
                        phase_td_prev = best_chi2pb_rT
                    elif phase_name == 'chi2_dphi':
                        phase_td_prev = best_chi2pb_d
                    elif phase_name == 'chi2_total':
                        phase_td_prev = best_chi2pb_rT + best_chi2pb_d
                    elif phase_name == 'td_rT':
                        phase_td_prev = td_rT_current
                    elif phase_name == 'td_total':
                        phase_td_prev = td_current
                    else:
                        phase_td_prev = td_current
                    print(f"\n  Phase {phase_idx+1}: {phase_name}")
                    continue
                else:
                    print(f"\n  All phases exhausted — stopping")
                    break
        else:
            # Hard stop: a component is now worse than the prior baseline by >50%
            if rT_degraded or d_degraded:
                print(f"\n  Stopping: component TD exceeds baseline by >50%.")
                if rT_degraded:
                    print(f"    rT: {td_rT_0:.1f} (baseline) → {best_td_rT:.1f} ({rT_vs_base:+.0f}%)")
                if d_degraded:
                    print(f"    dphi: {td_d_0:.1f} (baseline) → {best_td_d:.1f} ({d_vs_base:+.0f}%)")
                print(f"  This prior likely requires corrections incompatible with simultaneous")
                print(f"  improvement in both observables for this theory accuracy.")
                selection_log.pop()
                break
            elif metric_drop < 0:
                print(f"  Total {focus} got worse — stopping")
                selection_log.pop()
                break
            elif metric_pct_step < min_improvement_pct and step > 0:
                print(f"\n  Stopping: step {focus} improvement {metric_pct_step:.2f}% < {min_improvement_pct}%")
                selected.append(best_idx)
                available.remove(best_idx)
                break

        selected.append(best_idx)
        available.remove(best_idx)
        dual_current = best_dual
        td_current = best_td
        td_rT_current = best_td_rT
        td_d_current = best_td_d
        metric_current = best_metric
        best_lam = best_lam_new
        # Update running best per component
        td_rT_best = min(td_rT_best, best_td_rT)
        td_d_best  = min(td_d_best,  best_td_d)
        if select_strategy in ('rT_then_dphi', 'chi2_phased', 'td_phased'):
            phase_td_prev = phase_td_new

        # -- Backward step: try removing each existing moment --
        backward_removed = True
        while backward_removed and len(selected) > 1 and not no_backward:
            backward_removed = False
            best_back_td = td_current
            best_back_idx = None
            best_back_lam = None
            best_back_pos = None

            for k, idx in enumerate(selected):
                if idx in _forced_set:  # never remove forced moments
                    continue
                trial_set = [x for x in selected if x != idx]
                pairs_t = all_pairs[trial_set]
                targets_t = all_targets[trial_set]
                sigmas_t = [all_sigmas[i_] for i_ in trial_set]

                warm = np.delete(best_lam, k) if len(best_lam) > 0 else None

                with contextlib.redirect_stdout(io.StringIO()):
                    model_b = MaxEntDual(F_sub, G_sub, pairs_t, targets_t, w_sub,
                                         sigmas_target=sigmas_t,
                                         F_names=F_names, G_names=G_names)
                    if warm is not None and len(warm) > 0:
                        model_b.lam[:len(warm)] = warm
                    optimize_newton(model_b, max_steps=newton_steps,
                                    tol=newton_tol, verbose=False)

                w_b = model_b.get_weights()
                w_b = w_b / w_b.sum()
                td_rT_b, td_d_b, _, _ = compute_td_split(prior_data, w_b, target_dists, n_ev)
                td_b = td_rT_b + td_d_b

                if td_b < best_back_td:
                    best_back_td = td_b
                    best_back_idx = idx
                    best_back_lam = model_b.lam.copy()
                    best_back_pos = k

            if best_back_idx is not None:
                i_r, j_r = all_pairs[best_back_idx]
                rm_name = _display_name(F_names[i_r], G_names[j_r])
                selected.remove(best_back_idx)
                available.append(best_back_idx)
                td_current = best_back_td
                best_lam = best_back_lam
                dual_current = 0  # not tracked precisely after backward
                backward_removed = True
                td_pct_back = 100 * (1 - td_current / td_baseline)
                print(f"      ← removed {rm_name:<25}  TD={td_current:.1f} [{td_pct_back:+.1f}%]")

    # -- Summary --
    final_entry = selection_log[-1] if selection_log else None
    print(f"\n{'='*80}")
    print(f"[Selection Summary: {len(selected)} moments]")
    print(f"{'='*80}")
    print(f"  Prior TD:    {td_baseline:.4f}  (rT={td_rT_0:.4f}, dphi={td_d_0:.4f})")
    print(f"  Prior dual:  {dual_baseline:.6f}")
    if final_entry:
        print(f"  Final TD:    {final_entry['td']:.4f}  "
              f"(rT={final_entry['td_rT']:.4f}, dphi={final_entry['td_dphi']:.4f})")
        print(f"  Final dual:  {final_entry['dual']:.6f}")
        total_td_pct = 100 * (1 - final_entry['td'] / td_baseline)
        total_dual_drop = dual_baseline - final_entry['dual']
        print(f"  TD improvement:   {total_td_pct:.1f}%")
        print(f"  Dual improvement: {total_dual_drop:.4e}")

    print(f"\n  Step-by-step:")
    print(f"  {'#':<4} {'Moment':<28} {'Pw':>3} {'Dual':>10} {'dDual':>10} "
          f"{'TD':>8} {'TD%':>7} {'RMS pull':>9} {'N_eff':>7}")
    print(f"  {'-'*100}")
    for e in selection_log:
        print(f"  {e['step']:<4} {e['name']:<28} {e['power']:>3} "
              f"{e['dual']:10.4f} {e['delta_dual']:+10.4e} "
              f"{e['td']:8.1f} "
              f"{e['pct_improvement']:+6.1f}% {e['rms_pull']:9.3f} {100*e['neff']:6.1f}%")

    # Clean up multiprocessing pool
    if _pool is not None:
        _pool.close()
        _pool.join()

    return selected, selection_log


def print_diagnostics(model, F_names, G_names, top_k=10):
    """Print moment matching quality"""
    _, m_rew, _ = model._compute_logZ_moments_cov(model.lam, need_cov=False)

    m_prior = model._ensure_prior_moments() if hasattr(model, '_ensure_prior_moments') else model.m_prior
    targets = model.targets
    sigma = model.sigma

    diff = m_rew - targets
    pull_prior = (m_prior - targets) / np.maximum(sigma, 1e-30)
    pull_rew = diff / np.maximum(sigma, 1e-30)
    den = np.maximum(np.abs(targets), 0.1 * sigma)
    pct_err_prior = 100 * (m_prior - targets) / den
    pct_err_rew = 100 * diff / den

    idx_sort = np.argsort(np.abs(pull_rew))[::-1]

    print(f"\n{'Moment':<25} {'Target':>11} {'Prior':>11} {'Prior %Δ':>9} {'Reweighted':>11} {'Rew %Δ':>9} {'Pull':>7} {'σ_target':>11}")
    print("="*120)

    for i in idx_sort[:top_k]:
        i_f, j_g = model.pairs[i]
        name = _display_name(F_names[i_f], G_names[j_g])

        print(f"{name:<25} {targets[i]:11.5e} {m_prior[i]:11.5e} {pct_err_prior[i]:+8.2f}% "
              f"{m_rew[i]:11.5e} {pct_err_rew[i]:+8.2f}% {pull_rew[i]:+6.2f} {sigma[i]:11.3e}")

    print("\n" + "="*120)
    print(f"{'Summary:':<25} {'Mean |%Δ|':<20} {'Max |%Δ|':<20} {'RMS Pull':<20}")
    print(f"{'  Prior:':<25} {np.mean(np.abs(pct_err_prior)):19.2f}% {np.max(np.abs(pct_err_prior)):19.2f}% {np.sqrt(np.mean(pull_prior**2)):19.2f}")
    print(f"{'  Reweighted:':<25} {np.mean(np.abs(pct_err_rew)):19.2f}% {np.max(np.abs(pct_err_rew)):19.2f}% {np.sqrt(np.mean(pull_rew**2)):19.2f}")


# ========================================
# Load Target Distributions (with uncertainties)
# ========================================
# ATLAS-style variable-width bin edges (fine at peak, coarser in tail).
# Matches resolution shown in run-mode rT plots (~45 bins each).
ATLAS_RT_EDGES = np.unique(np.concatenate([
    np.array([0.01]),              # start at 0.01 (theory's effective resolution)
    np.arange(0.025, 0.50, 0.025), # fine peak
    np.arange(0.50, 1.00, 0.05),   # 10 bins
    np.arange(1.00, 2.00, 0.10),   # 10 bins
    np.arange(2.00, 3.00, 0.25),   # 4 bins
    np.array([3.00, 4.00, 5.00]),  # 2 bins
]))
ATLAS_DPHI_EDGES = np.unique(np.concatenate([
    np.arange(0.0, 0.50, 0.025),   # 20 bins, fine near back-to-back peak
    np.arange(0.50, 1.00, 0.05),   # 10 bins
    np.arange(1.00, 2.00, 0.10),   # 10 bins
    np.arange(2.00, np.pi, 0.20),  # ~6 bins
    np.array([np.pi]),
]))

# Log-uniform edges starting at 0.01 (theory's effective resolution limit).
# Events with rT < 0.01 are excluded from the metric.
LOG_RT_EDGES = 10 ** np.linspace(np.log10(0.01), np.log10(5.0), 28)  # 27 bins, 0.01 to 5
LOG_DPHI_EDGES = 10 ** np.linspace(np.log10(0.01), np.log10(np.pi), 26)  # 25 bins, 0.01 to π

# Coarse variants for low-statistics runs (~1M events)
LOG_RT_EDGES_COARSE = 10 ** np.linspace(np.log10(0.01), np.log10(5.0), 16)  # 15 bins
LOG_DPHI_EDGES_COARSE = 10 ** np.linspace(np.log10(0.01), np.log10(np.pi), 14)  # 13 bins
ATLAS_RT_EDGES_COARSE = np.unique(np.concatenate([
    np.array([0.01]),             # start at 0.01 (match regular ATLAS)
    np.arange(0.05, 0.50, 0.05),  # peak
    np.arange(0.50, 1.00, 0.10),  # 5 bins
    np.arange(1.00, 2.00, 0.25),  # 4 bins
    np.array([2.0, 2.5, 3.0, 4.0, 5.0]),  # 4 bins
]))
ATLAS_DPHI_EDGES_COARSE = np.unique(np.concatenate([
    np.arange(0.0, 0.50, 0.05),   # 10 bins
    np.arange(0.50, 1.00, 0.10),  # 5 bins
    np.arange(1.00, 2.00, 0.25),  # 4 bins
    np.array([2.0, 2.5, np.pi]),  # 2 bins
]))


def rebin_target_dist(td, new_edges):
    """Aggregate a target_dists entry onto coarser bin edges.

    Density rebinned by integral: new_dens[i] = (Σ_j old_dens[j] * overlap[i,j]) / new_width[i]
    Stat unc combined in quadrature: new_unc[i] = sqrt(Σ_j (old_unc[j] * overlap[i,j])^2) / new_width[i]
    """
    old_edges = td['edges']
    old_widths = np.diff(old_edges)
    new_widths = np.diff(new_edges)
    n_new = len(new_widths)
    n_old = len(old_widths)

    # Build overlap matrix (n_new, n_old)
    a = new_edges[:-1][:, None]  # (n_new, 1)
    b = new_edges[1:][:, None]
    oa = old_edges[:-1][None, :]  # (1, n_old)
    ob = old_edges[1:][None, :]
    overlap = np.maximum(0, np.minimum(b, ob) - np.maximum(a, oa))  # (n_new, n_old)

    def rebin_dens(dens):
        return (overlap @ dens) / new_widths

    def rebin_unc(unc):
        # Variance of integrated count over overlap = (unc * overlap)^2
        # (assuming uncorrelated within original bins).
        return np.sqrt(overlap**2 @ (unc**2)) / new_widths

    out = dict(td)
    out['edges'] = new_edges
    out['central'] = rebin_dens(td['central'])
    out['central_unc'] = rebin_unc(td['central_unc'])
    if 'min' in td:
        out['min'] = rebin_dens(td['min'])
    if 'max' in td:
        out['max'] = rebin_dens(td['max'])
    if 'var_map' in td:
        out['var_map'] = {k: rebin_dens(v) for k, v in td['var_map'].items()}
    if 'var_unc_map' in td:
        out['var_unc_map'] = {k: rebin_unc(v) for k, v in td['var_unc_map'].items()}
    return out


def load_target_distributions(moments_csv, acc):
    """Load target histograms with statistical uncertainties."""
    if moments_csv is None:
        return {}

    import csv
    from collections import defaultdict

    if isinstance(moments_csv, (list, tuple)):
        csv_files = list(moments_csv)
    else:
        csv_files = [moments_csv]

    rows = []
    for path in csv_files:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = {
                        'dist': row['dist'],
                        'acc': row['acc'],
                        'fo': row['ScaleFO'],
                        'res': row['ScaleRes'],
                        'bin_lo': float(row['bin_lo']),
                        'bin_hi': float(row['bin_hi']),
                        'count': float(row.get('count', 'nan')) if 'count' in row else np.nan,
                        'density': float(row.get('density', 'nan')) if 'density' in row else np.nan,
                        'uncertainty': float(row.get('uncertainty', 'nan')) if 'uncertainty' in row else np.nan,
                    }
                    rows.append(r)
                except Exception:
                    continue

    if not rows:
        return {}

    acc_variants = {acc, acc.replace("'", "p"), acc.replace("'", ""),
                     acc.replace("p", "'")}
    acc_rows = [r for r in rows if r['acc'] in acc_variants]

    if not acc_rows:
        return {}

    print(f"      Found {len(acc_rows)} rows for accuracy ~ '{acc}'")
    by_dist = defaultdict(lambda: defaultdict(list))
    for r in acc_rows:
        by_dist[r['dist']][(r['fo'], r['res'])].append(r)

    result = {}
    for dist_name in ['dphiDist', 'rTDist']:
        if dist_name not in by_dist:
            continue

        central_key = None
        for (fo, res) in by_dist[dist_name].keys():
            if str(fo).upper().startswith('CV') and str(res).upper().startswith('CV'):
                central_key = (fo, res)
                break
        if central_key is None:
            continue

        central_rows = sorted(by_dist[dist_name][central_key], key=lambda r: r['bin_lo'])
        edges = np.array([central_rows[0]['bin_lo']] + [r['bin_hi'] for r in central_rows], dtype=float)
        widths = np.diff(edges)

        central_dens = np.array([
            (r['density'] if np.isfinite(r['density']) else r['count'] / max(bw, 1e-300))
            for r, bw in zip(central_rows, widths)
        ], dtype=float)

        central_unc = np.array([
            r['uncertainty'] if np.isfinite(r['uncertainty']) else 0.0
            for r in central_rows
        ], dtype=float)

        var_densities = []
        var_density_map = {central_key: central_dens}
        var_unc_map = {central_key: central_unc}

        for key, var_rows in by_dist[dist_name].items():
            if key == central_key:
                continue
            var_rows_sorted = sorted(var_rows, key=lambda r: r['bin_lo'])
            var_edges = np.array([var_rows_sorted[0]['bin_lo']] + [r['bin_hi'] for r in var_rows_sorted], dtype=float)

            if len(var_edges) == len(edges) and np.allclose(var_edges, edges):
                var_dens = np.array([
                    (r['density'] if np.isfinite(r['density']) else r['count'] / max(bw, 1e-300))
                    for r, bw in zip(var_rows_sorted, widths)
                ], dtype=float)
                var_unc = np.array([
                    r['uncertainty'] if np.isfinite(r['uncertainty']) else 0.0
                    for r in var_rows_sorted
                ], dtype=float)

                var_densities.append(var_dens)
                var_density_map[key] = var_dens
                var_unc_map[key] = var_unc

        if var_densities:
            var_array = np.vstack(var_densities)
            dens_min = np.min(var_array, axis=0)
            dens_max = np.max(var_array, axis=0)
        else:
            dens_min = central_dens.copy()
            dens_max = central_dens.copy()

        result[dist_name] = {
            'edges': edges,
            'central': central_dens,
            'central_unc': central_unc,
            'min': dens_min,
            'max': dens_max,
            'var_map': var_density_map,
            'var_unc_map': var_unc_map,
            'n_variations': len(var_densities)
        }

        rel_unc = 100 * central_unc / np.maximum(central_dens, 1e-300)
        print(f"      ✓ {dist_name}: {len(edges)-1} bins, {len(var_densities)} scale vars, "
              f"stat unc: {np.median(rel_unc):.1f}% median")

    return result


def load_pT_theory(qT_file, acc):
    """Parse Wan-Li qT theory .m file into a target-dist-style dict for pT.

    Returns dict with edges/central/central_unc/min/max/var_map, matching the
    structure of load_target_distributions entries so the rT-style multi-panel
    plotting can be reused for pT. var_map keys are (res, fo) tuples (excluding
    central) so _classify_variation groups them into resum/fo/cs/kappa.
    """
    if not qT_file or not os.path.exists(qT_file):
        return None
    text = open(qT_file).read()
    pattern = (r'qTMat\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]\s*'
               r'=\s*\{((?:\{[^}]+\},?\s*)+)\}')
    data = {}
    for m in re.finditer(pattern, text):
        a, res, fo = m.group(1), m.group(2), m.group(3)
        rows = re.findall(r'\{([^}]+)\}', m.group(4))
        bins = [[float(x) for x in r.split(',')[:4]] for r in rows]
        data[(a, res, fo)] = np.array(bins)

    # accuracy matching (handle p<->' variants)
    acc_variants = {acc, acc.replace("'", "p"), acc.replace("'", ""),
                    acc.replace("p", "'")}
    central = None
    for (a, res, fo), bins in data.items():
        if a in acc_variants and res.upper().startswith('CV') and fo.upper().startswith('CV'):
            central = (a, res, fo, bins)
            break
    if central is None:
        return None
    a0, _, _, cb = central
    edges = np.append(cb[:, 0], cb[-1, 1])
    cen = cb[:, 2]
    unc = cb[:, 3]

    var_map = {}
    for (a, res, fo), bins in data.items():
        if a != a0:
            continue
        if res.upper().startswith('CV') and fo.upper().startswith('CV'):
            continue  # skip central
        if len(bins) != len(cen):
            continue
        var_map[(res, fo)] = bins[:, 2]

    if var_map:
        arr = np.vstack(list(var_map.values()))
        dmin = np.minimum(np.min(arr, axis=0), cen)
        dmax = np.maximum(np.max(arr, axis=0), cen)
    else:
        dmin = cen.copy(); dmax = cen.copy()

    return {'edges': edges, 'central': cen, 'central_unc': unc,
            'min': dmin, 'max': dmax, 'var_map': var_map,
            'n_variations': len(var_map)}


# ========================================
# Plotting
# ========================================
def hist_to_density(x, w, edges):
    x = np.asarray(x)
    w = np.asarray(w)
    mask = (x >= edges[0]) & (x < edges[-1]) & np.isfinite(x) & np.isfinite(w)
    counts, _ = np.histogram(x[mask], bins=edges, weights=w[mask])
    widths = np.diff(edges)
    return counts / np.maximum(widths, 1e-300)


def match_area(dens, ref_dens, edges, label=None):
    widths = np.diff(edges)
    A_ref = float(np.sum(ref_dens * widths))
    A = float(np.sum(dens * widths))
    if A > 0:
        ratio = A_ref / A
        if label:
            print(f"    match_area [{label}]: MC={A:.4f}, Target={A_ref:.4f}, scale={ratio:.4f}")
        dens = dens * ratio
    return dens


def precompute_variation_hists(w_var, hist_edges):
    """Histogram a weight array for all observables, return small dict of densities."""
    result = {}
    for obs_name, (x, edges) in hist_edges.items():
        result[obs_name] = hist_to_density(x, w_var, edges)
    return result


def rebin_density(dens, edges, factor=3):
    """Rebin density array by merging `factor` adjacent bins.
    Returns new (density, edges) with proper area-weighted averaging."""
    n = len(dens)
    n_new = n // factor
    new_edges = np.empty(n_new + 1)
    new_dens = np.empty(n_new)
    widths = np.diff(edges)
    for i in range(n_new):
        i0 = i * factor
        i1 = min(i0 + factor, n)
        new_edges[i] = edges[i0]
        # Area-weighted: sum(dens_j * w_j) / sum(w_j)
        new_dens[i] = np.sum(dens[i0:i1] * widths[i0:i1]) / np.sum(widths[i0:i1])
    new_edges[-1] = edges[min(n_new * factor, n)]
    return new_dens, new_edges


def adaptive_rebin_edges(t_cen, t_unc, edges, max_rel_unc=0.10, min_bins=15):
    """Compute adaptive bin grouping from target statistics.

    Merges adjacent bins until each merged bin has relative uncertainty
    below max_rel_unc. Returns list of (i_start, i_end) slices defining
    the new bins in terms of the original bin indices.

    Parameters:
      t_cen: (N,) target central densities
      t_unc: (N,) target stat uncertainties on density
      edges: (N+1,) bin edges
      max_rel_unc: merge bins until rel unc < this (default 10%)
      min_bins: minimum number of output bins (relax threshold if needed)
    """
    widths = np.diff(edges)
    n = len(t_cen)
    groups = []
    i = 0
    while i < n:
        i0 = i
        # Accumulate bins
        cum_area = 0.0
        cum_unc2 = 0.0
        while i < n:
            cum_area += t_cen[i] * widths[i]
            cum_unc2 += (t_unc[i] * widths[i])**2
            i += 1
            # Check if merged bin has good enough stats
            cum_width = edges[i] - edges[i0]
            merged_dens = cum_area / cum_width if cum_width > 0 else 0.0
            merged_unc = np.sqrt(cum_unc2) / cum_width if cum_width > 0 else 0.0
            if merged_dens > 0 and merged_unc / merged_dens < max_rel_unc:
                break
        groups.append((i0, i))

    # If too few bins, relax and redo with larger threshold
    if len(groups) < min_bins and max_rel_unc < 0.5:
        return adaptive_rebin_edges(t_cen, t_unc, edges,
                                    max_rel_unc=max_rel_unc * 1.5,
                                    min_bins=min_bins)
    return groups


def adaptive_rebin_density(dens, edges, groups):
    """Rebin a density array using pre-computed adaptive bin groups.

    Parameters:
      dens: (N,) density values
      edges: (N+1,) original bin edges
      groups: list of (i_start, i_end) from adaptive_rebin_edges

    Returns: (new_dens, new_edges)
    """
    widths = np.diff(edges)
    new_dens = np.empty(len(groups))
    new_edges = np.empty(len(groups) + 1)
    for k, (i0, i1) in enumerate(groups):
        new_edges[k] = edges[i0]
        w_slice = widths[i0:i1]
        d_slice = dens[i0:i1]
        new_dens[k] = np.sum(d_slice * w_slice) / np.sum(w_slice)
    new_edges[-1] = edges[groups[-1][1]]
    return new_dens, new_edges


def plot_distributions(prior_data, w_prior, w_rew, acc, output_dir, moments_csv,
                       reweighted_dict=None, stat_band_weights=None, rebin_factor=3,
                       pT_theory_file=None):
    """RIVET-style publication plots with scale/stat bands.

    stat_band_weights: tuple (w_rew_up, w_rew_down) from fitting target +/- sigma_stat
    reweighted_dict: either {key: w_array} (legacy) or {key: {'rT': density, ...}} (precomputed)
    """
    import matplotlib as mpl
    from matplotlib.ticker import AutoMinorLocator

    # -- RIVET style --
    # Check if LaTeX is available
    import shutil
    use_tex = shutil.which('latex') is not None
    mpl.rcParams.update({
        'text.usetex': use_tex,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.frameon': False,
        'legend.fontsize': 9,
        'legend.handlelength': 1.5,
        'lines.linewidth': 1.2,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
    })

    C = {
        'target': '#000000', 'target_stat': '#999999',
        'target_scale': '#4682B4', 'prior': '#D62728',
        'rew': '#1F77B4', 'rew_env': '#2CA02C',
    }

    # Grouped variation band colors and labels
    VARIATION_GROUPS = {
        'resum':  {'color': '#4682B4', 'label': r'Resum.\ scales'},
        'cs':     {'color': '#D95F02', 'label': r'CS kernel ($C_0^{\mathrm{NP}}$)'},
        'kappa':  {'color': '#7570B3', 'label': r'$\kappa_{\mathrm{NP}}$'},
        'fo':     {'color': '#1B9E77', 'label': r'FO scales'},
    }

    def _classify_variation(scale_key):
        """Classify a variation scale key into one of the 4 groups."""
        if isinstance(scale_key, tuple):
            s = '/'.join(str(x) for x in scale_key)
        else:
            s = str(scale_key)
        su = s.upper()
        if 'CV->FO/CV->RES' in su or su == 'CENTRAL':
            return None  # central, skip
        if 'C0_NP' in su:
            return 'cs'
        if 'KAPPA_NP' in su:
            return 'kappa'
        # FO scales: MuR, MuF, MuRF only on the FO side (not CV->FO)
        if any(x in su for x in ['MUR->FO', 'MUF->FO', 'MURF->FO']):
            return 'fo'
        # Everything else is resummation
        return 'resum'

    def _build_grouped_envelopes(obs_key, x_data, edges, ref_density,
                                 variation_dict, area_match_target=None,
                                 var_map=None):
        """Build per-group envelopes from variation dict.

        Returns dict: group_name -> (lo, hi, n_vars)
        """
        groups = {}  # group -> list of density arrays
        if not variation_dict:
            return {}
        for scale_key, val in variation_dict.items():
            grp = _classify_variation(scale_key)
            if grp is None:
                continue
            if isinstance(val, dict):
                d_v = val.get(obs_key)
                if d_v is None:
                    continue
            else:
                d_v = hist_to_density(x_data, val, edges)
            d_v = np.where(ref_density > 0, d_v, 0.0)
            if area_match_target is not None:
                # For reweighted: match area to per-variation target if available
                if var_map is not None:
                    t_var = var_map.get(
                        scale_key[:2] if len(scale_key) > 2 else scale_key,
                        area_match_target)
                else:
                    t_var = area_match_target
                d_v = match_area(d_v, t_var, edges)
            else:
                # For target: already densities, no area matching needed
                pass
            groups.setdefault(grp, []).append(d_v)

        result = {}
        for grp, densities in groups.items():
            arr = np.vstack(densities)
            result[grp] = (np.min(arr, axis=0), np.max(arr, axis=0), len(densities))
        return result

    def _build_target_grouped_envelopes(obs_key, var_map):
        """Build per-group envelopes from target var_map."""
        if not var_map:
            return {}
        groups = {}
        for scale_key, dens in var_map.items():
            grp = _classify_variation(scale_key)
            if grp is None:
                continue
            groups.setdefault(grp, []).append(dens)
        result = {}
        for grp, densities in groups.items():
            arr = np.vstack(densities)
            result[grp] = (np.min(arr, axis=0), np.max(arr, axis=0), len(densities))
        return result

    def _step_xy(edges, vals):
        x = np.repeat(edges, 2)[1:-1]
        y = np.repeat(vals, 2)
        return x, y

    def _step_fill(ax_, edges, lo, hi, **kw):
        x_lo, y_lo = _step_xy(edges, lo)
        _, y_hi = _step_xy(edges, hi)
        ax_.fill_between(x_lo, y_lo, y_hi, **kw)

    def _step_line(ax_, edges, vals, **kw):
        x, y = _step_xy(edges, vals)
        ax_.plot(x, y, **kw)

    def _make_figure():
        fig_, (ax_, axr_) = plt.subplots(
            2, 1, figsize=(6.5, 7.5), height_ratios=[3, 1], sharex=True,
            gridspec_kw={'hspace': 0.0})
        return fig_, ax_, axr_

    def _finish_axes(ax_, axr_, xlabel, logy=True, ratio_range=(0.85, 1.15),
                     ratio_label=r'Ratio to theory'):
        if logy:
            ax_.set_yscale('log')
        ax_.xaxis.set_minor_locator(AutoMinorLocator())
        ax_.legend(loc='upper right', ncol=1)

        acc_label = acc.replace("'", r"$'$").replace("+", r"$+$")
        ax_.text(0.04, 0.96, r'$pp \to Z/\gamma^* \to \ell\ell$',
                 transform=ax_.transAxes, va='top', ha='left', fontsize=10)
        ax_.text(0.04, 0.89, acc_label,
                 transform=ax_.transAxes, va='top', ha='left', fontsize=9,
                 color='#555555')

        axr_.set_ylabel(ratio_label)
        axr_.set_xlabel(xlabel)
        axr_.set_ylim(*ratio_range)
        axr_.xaxis.set_minor_locator(AutoMinorLocator())
        axr_.yaxis.set_minor_locator(AutoMinorLocator())

    def _save(fig_, output_dir_, name_, acc_slug_):
        for ext in ['pdf', 'png']:
            p = f"{output_dir_}/{name_}_{acc_slug_}.{ext}"
            fig_.savefig(p, dpi=300 if ext == 'pdf' else 200)
        plt.close(fig_)
        print(f"  Saved: {output_dir_}/{name_}_{acc_slug_}.pdf (.png)")

    # -- Load targets --
    target = {}
    print(f"\n  [Loading Target Distributions]")
    if moments_csv:
        target = load_target_distributions(moments_csv, acc)

    # Optional pT theory (Wan-Li qT) → target-dist-style entry for fancy pT plot
    if pT_theory_file:
        pT_t = load_pT_theory(pT_theory_file, acc)
        if pT_t is not None:
            target['pTDist'] = pT_t
            print(f"      ✓ pTDist (theory qT): {len(pT_t['edges'])-1} bins, "
                  f"{pT_t['n_variations']} scale vars")
        else:
            print(f"      ⚠ pT theory file not parsed: {pT_theory_file}")

    acc_slug = acc.replace("'", "p").replace(" ", "")

    def _build_rew_envelope(obs_key, x_data, edges, t_cen_ref, reweighted_dict_, var_map_):
        if not reweighted_dict_:
            return None, None, 0
        rew_densities_ = []
        for scale_key_, val_ in reweighted_dict_.items():
            if isinstance(val_, dict):
                d_v_ = val_.get(obs_key)
                if d_v_ is None:
                    continue
            else:
                d_v_ = hist_to_density(x_data, val_, edges)
            d_v_ = np.where(t_cen_ref > 0, d_v_, 0.0)
            t_var_ = var_map_.get(scale_key_[:2] if len(scale_key_) > 2 else scale_key_,
                                  t_cen_ref)
            d_v_ = match_area(d_v_, t_var_, edges)
            rew_densities_.append(d_v_)
        if not rew_densities_:
            return None, None, 0
        arr_ = np.vstack(rew_densities_)
        return np.min(arr_, axis=0), np.max(arr_, axis=0), len(rew_densities_)

    def _build_rew_envelope_notarget(obs_key, x_data, edges, d_rew_cen_, reweighted_dict_):
        if not reweighted_dict_:
            return None, None, 0
        rew_densities_ = []
        for scale_key_, val_ in reweighted_dict_.items():
            if isinstance(val_, dict):
                d_v_ = val_.get(obs_key)
                if d_v_ is None:
                    continue
            else:
                d_v_ = hist_to_density(x_data, val_, edges)
            d_v_ = match_area(d_v_, d_rew_cen_, edges)
            rew_densities_.append(d_v_)
        if not rew_densities_:
            return None, None, 0
        arr_ = np.vstack(rew_densities_)
        return np.min(arr_, axis=0), np.max(arr_, axis=0), len(rew_densities_)

    # ================================================================
    # PLOT: rT distribution (with theory target)
    # ================================================================
    td_prior_rT, td_rew_rT = 0, 0
    if 'rTDist' in target:
        edges = target['rTDist']['edges']
        t_cen = target['rTDist']['central']
        t_unc = target['rTDist']['central_unc']
        t_min = target['rTDist']['min']
        t_max = target['rTDist']['max']
        var_map = target['rTDist'].get('var_map', {})

        d_prior = hist_to_density(prior_data['rT'], w_prior, edges)
        d_rew = hist_to_density(prior_data['rT'], w_rew, edges)
        mask = t_cen > 0
        d_prior = np.where(mask, d_prior, 0.0)
        d_rew = np.where(mask, d_rew, 0.0)
        d_prior = match_area(d_prior, t_cen, edges, label='rT prior')
        d_rew = match_area(d_rew, t_cen, edges, label='rT reweighted')

        td_prior_rT = triangle_divergence(d_prior, t_cen)
        td_rew_rT = triangle_divergence(d_rew, t_cen)
        chi2d_prior_rT = chi2_divergence(d_prior, t_cen)
        chi2d_rew_rT = chi2_divergence(d_rew, t_cen)
        chi2b_prior_rT, nbins_rT = chi2_per_bin(d_prior, t_cen, t_unc)
        chi2b_rew_rT, _ = chi2_per_bin(d_rew, t_cen, t_unc)
        print(f"  [TD] rT: Prior->Target = {td_prior_rT:.6e}, Rew->Target = {td_rew_rT:.6e}")
        print(f"  [χ²] rT: Prior = {chi2d_prior_rT:.6e}, Rew = {chi2d_rew_rT:.6e}")
        print(f"  [χ²/bin] rT: Prior = {chi2b_prior_rT:.2f}, Rew = {chi2b_rew_rT:.2f}  ({nbins_rT} bins)")

        # Build grouped envelopes for reweighted
        rew_groups = _build_grouped_envelopes(
            'rT', prior_data['rT'], edges, t_cen, reweighted_dict,
            area_match_target=t_cen, var_map=var_map)

        # Build grouped envelopes for theory target
        tgt_groups = _build_target_grouped_envelopes('rTDist', var_map)

        # ATLAS-like variable-width binning for plotting (ATLAS Z pT 1912.02844
        # bin structure scaled to rT = pT/m_Z). Metrics still on native fine bins.
        edges_orig = edges.copy()
        target_edges = np.unique(np.concatenate([
            np.arange(0.0,  0.50, 0.025),     # fine at peak (20 bins of 0.025)
            np.arange(0.50, 1.00, 0.05),      # 10 bins of 0.05
            np.arange(1.00, 2.00, 0.10),      # 10 bins of 0.10
            np.arange(2.00, 3.00, 0.25),      # 4 bins of 0.25
            np.array([3.00, 4.00, 5.00]),     # 2 bins of 1.0
        ]))
        # Snap to nearest native edge
        target_edges = np.array([edges_orig[np.abs(edges_orig - e).argmin()]
                                 for e in target_edges])
        target_edges = np.unique(target_edges)
        # Build bin-merge groups (i_start, i_end) into native bins
        rb_groups = []
        for k in range(len(target_edges) - 1):
            i0 = np.searchsorted(edges_orig, target_edges[k], side='left')
            i1 = np.searchsorted(edges_orig, target_edges[k+1], side='left')
            if i1 > i0:
                rb_groups.append((i0, i1))
        t_cen, edges = adaptive_rebin_density(t_cen, edges_orig, rb_groups)
        t_unc, _ = adaptive_rebin_density(t_unc, edges_orig, rb_groups)
        d_prior, _ = adaptive_rebin_density(d_prior, edges_orig, rb_groups)
        d_rew, _ = adaptive_rebin_density(d_rew, edges_orig, rb_groups)
        mask = t_cen > 0

        for grp in list(rew_groups.keys()):
            lo, hi, n = rew_groups[grp]
            lo, _ = adaptive_rebin_density(lo, edges_orig, rb_groups)
            hi, _ = adaptive_rebin_density(hi, edges_orig, rb_groups)
            rew_groups[grp] = (lo, hi, n)
        for grp in list(tgt_groups.keys()):
            lo, hi, n = tgt_groups[grp]
            lo, _ = adaptive_rebin_density(lo, edges_orig, rb_groups)
            hi, _ = adaptive_rebin_density(hi, edges_orig, rb_groups)
            tgt_groups[grp] = (lo, hi, n)

        # Total envelopes (already rebinned)
        tgt_total_lo = t_cen.copy()
        tgt_total_hi = t_cen.copy()
        for grp, (lo, hi, n) in tgt_groups.items():
            tgt_total_lo = np.minimum(tgt_total_lo, lo)
            tgt_total_hi = np.maximum(tgt_total_hi, hi)
        rew_total_lo = d_rew.copy()
        rew_total_hi = d_rew.copy()
        for grp, (lo, hi, n) in rew_groups.items():
            rew_total_lo = np.minimum(rew_total_lo, lo)
            rew_total_hi = np.maximum(rew_total_hi, hi)

        # ── Multi-panel figure ──
        sub_groups = ['resum', 'fo', 'cs', 'kappa']
        sub_labels = {
            'resum': r'Resum.',
            'fo': r'FO',
            'cs': r'$C_0^{\mathrm{NP}}$',
            'kappa': r'$\kappa_{\mathrm{NP}}$',
        }
        n_sub = len(sub_groups)
        height_ratios = [3, 1.2] + [0.8] * n_sub
        fig, axes = plt.subplots(
            2 + n_sub, 1, figsize=(6.5, 10),
            height_ratios=height_ratios, sharex=True,
            gridspec_kw={'hspace': 0.0})
        ax = axes[0]
        axr = axes[1]

        # ── Extended MC histograms for upper panel (rT up to 2.0) ──
        rT_ext_max = max(2.0, float(edges[-1]))
        bin_width_ext = edges[1] - edges[0]  # use same bin width as rebinned
        ext_edges = np.arange(edges[-1], rT_ext_max + bin_width_ext, bin_width_ext)
        if len(ext_edges) > 1:
            edges_full = np.concatenate([edges, ext_edges[1:]])
            # Area normalization factor from theory range
            widths_th = np.diff(edges)
            A_ref = float(np.sum(t_cen * widths_th))
            # Prior extended
            d_prior_full_raw = hist_to_density(prior_data['rT'], w_prior, edges_full)
            A_prior_full = float(np.sum(d_prior_full_raw[:len(t_cen)] * widths_th))
            if A_prior_full > 0:
                d_prior_full = d_prior_full_raw * (A_ref / A_prior_full)
            else:
                d_prior_full = d_prior_full_raw
            # Rew extended
            d_rew_full_raw = hist_to_density(prior_data['rT'], w_rew, edges_full)
            A_rew_full = float(np.sum(d_rew_full_raw[:len(t_cen)] * widths_th))
            if A_rew_full > 0:
                d_rew_full = d_rew_full_raw * (A_ref / A_rew_full)
            else:
                d_rew_full = d_rew_full_raw
        else:
            edges_full = edges
            d_prior_full = d_prior
            d_rew_full = d_rew

        # ── Upper panel: clean, lines only ──
        _step_line(ax, edges_full, d_prior_full, color=C['prior'], lw=1.2, alpha=0.7,
                   label=r'Prior (Sherpa)', zorder=5)
        _step_line(ax, edges_full, d_rew_full, color=C['rew'], lw=1.6,
                   label=r'Reweighted', zorder=6)
        _step_line(ax, edges, t_cen, color=C['target'], lw=1.4,
                   label=r'Theory (central)', zorder=7)

        valid = t_cen[mask]
        if len(valid) > 0:
            ax.set_ylim(valid.min() * 0.3, valid.max() * 5)
        ax.set_yscale('log')
        ax.set_xlim(edges[0], rT_ext_max)
        ax.set_ylabel(r'$\mathrm{d}\sigma / \mathrm{d}r_T$')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper right', fontsize=9, ncol=1)

        acc_label = acc.replace("'", r"$'$").replace("+", r"$+$")
        ax.text(0.04, 0.96, r'$pp \to Z/\gamma^* \to \ell\ell$',
                transform=ax.transAxes, va='top', ha='left', fontsize=10)
        ax.text(0.04, 0.89, acc_label,
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                color='#555555')
        if td_prior_rT > 0:
            ax.text(0.04, 0.82,
                    rf'$\chi^2/\mathrm{{bin}}$: {chi2b_prior_rT:.1f} $\to$ {chi2b_rew_rT:.1f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    color='#555555')

        # ── Main ratio: total envelope ──
        with np.errstate(divide='ignore', invalid='ignore'):
            r_prior = np.where(mask, d_prior / t_cen, 1.0)
            r_rew = np.where(mask, d_rew / t_cen, 1.0)
            r_unc = np.where(mask, t_unc / t_cen, 0.0)
            r_tgt_lo = np.where(mask, tgt_total_lo / t_cen, 1.0)
            r_tgt_hi = np.where(mask, tgt_total_hi / t_cen, 1.0)
            r_rew_lo = np.where(mask, rew_total_lo / t_cen, 1.0)
            r_rew_hi = np.where(mask, rew_total_hi / t_cen, 1.0)

        _step_fill(axr, edges, r_tgt_lo, r_tgt_hi, color=C['target_scale'],
                   alpha=0.20, label=r'Theory unc.', zorder=1)
        _step_fill(axr, edges, 1 - r_unc, 1 + r_unc, color=C['target_stat'],
                   alpha=0.35, zorder=2)
        _step_fill(axr, edges, r_rew_lo, r_rew_hi, color=C['rew'],
                   alpha=0.15, label=r'Rew.\ unc.', zorder=3)
        axr.axhline(1.0, color='black', ls='-', lw=0.6, zorder=0)
        _step_line(axr, edges, r_prior, color=C['prior'], lw=1.2, alpha=0.7, zorder=5)
        _step_line(axr, edges, r_rew, color=C['rew'], lw=1.6, zorder=6)
        axr.set_ylabel(r'Ratio', fontsize=9)
        axr.set_ylim(0.85, 1.15)
        axr.xaxis.set_minor_locator(AutoMinorLocator())
        axr.yaxis.set_minor_locator(AutoMinorLocator())
        axr.legend(loc='upper right', fontsize=7, ncol=2)

        # ── Sub-ratio panels: one per uncertainty type ──
        for si, grp in enumerate(sub_groups):
            axs = axes[2 + si]
            gc = VARIATION_GROUPS[grp]

            # Theory band for this group
            if grp in tgt_groups:
                lo, hi, nv = tgt_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask, lo / t_cen, 1.0)
                    rhi = np.where(mask, hi / t_cen, 1.0)
                _step_fill(axs, edges, rlo, rhi, color=gc['color'],
                           alpha=0.20, zorder=1)

            # Reweighted band for this group
            if grp in rew_groups:
                lo, hi, nv = rew_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask, lo / t_cen, 1.0)
                    rhi = np.where(mask, hi / t_cen, 1.0)
                _step_fill(axs, edges, rlo, rhi, color=gc['color'],
                           alpha=0.35, hatch='///', linewidth=0, zorder=2)

            axs.axhline(1.0, color='black', ls='-', lw=0.5, zorder=0)
            _step_line(axs, edges, r_rew, color=C['rew'], lw=1.0, alpha=0.5, zorder=3)
            axs.set_ylim(0.92, 1.08)
            axs.yaxis.set_minor_locator(AutoMinorLocator())
            axs.set_ylabel(sub_labels[grp], fontsize=8, rotation=0,
                           labelpad=20, va='center')
            if si < n_sub - 1:
                axs.tick_params(labelbottom=False)

        axes[-1].set_xlabel(r'$r_T = p_T / m_{\ell\ell}$')
        axes[-1].xaxis.set_minor_locator(AutoMinorLocator())

        for ext in ['pdf', 'png']:
            p_ = f"{output_dir}/rT_{acc_slug}.{ext}"
            fig.savefig(p_, dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_dir}/rT_{acc_slug}.pdf (.png)")

    # ================================================================
    # PLOT: dphi distribution (with theory target)
    # ================================================================
    if 'dphiDist' in target:
        edges = target['dphiDist']['edges']
        t_cen = target['dphiDist']['central']
        t_unc = target['dphiDist']['central_unc']
        t_min = target['dphiDist']['min']
        t_max = target['dphiDist']['max']
        var_map = target['dphiDist'].get('var_map', {})

        d_prior = hist_to_density(prior_data['d'], w_prior, edges)
        d_rew = hist_to_density(prior_data['d'], w_rew, edges)
        mask = t_cen > 0
        d_prior = np.where(mask, d_prior, 0.0)
        d_rew = np.where(mask, d_rew, 0.0)
        d_prior = match_area(d_prior, t_cen, edges, label='dphi prior')
        d_rew = match_area(d_rew, t_cen, edges, label='dphi reweighted')

        td_prior_d = triangle_divergence(d_prior, t_cen)
        td_rew_d = triangle_divergence(d_rew, t_cen)
        chi2d_prior_d = chi2_divergence(d_prior, t_cen)
        chi2d_rew_d = chi2_divergence(d_rew, t_cen)
        chi2b_prior_d, nbins_d = chi2_per_bin(d_prior, t_cen, t_unc)
        chi2b_rew_d, _ = chi2_per_bin(d_rew, t_cen, t_unc)
        print(f"  [TD] dphi: Prior->Target = {td_prior_d:.6e}, Rew->Target = {td_rew_d:.6e}")
        print(f"  [χ²] dphi: Prior = {chi2d_prior_d:.6e}, Rew = {chi2d_rew_d:.6e}")
        print(f"  [χ²/bin] dphi: Prior = {chi2b_prior_d:.2f}, Rew = {chi2b_rew_d:.2f}  ({nbins_d} bins)")
        print(f"  [TD] TOTAL: Prior = {td_prior_rT + td_prior_d:.6e}, "
              f"Rew = {td_rew_rT + td_rew_d:.6e}")
        nb_tot = max(nbins_rT + nbins_d, 1)
        print(f"  [χ²/bin] TOTAL: Prior = "
              f"{(chi2b_prior_rT * nbins_rT + chi2b_prior_d * nbins_d) / nb_tot:.2f}, "
              f"Rew = "
              f"{(chi2b_rew_rT * nbins_rT + chi2b_rew_d * nbins_d) / nb_tot:.2f}  "
              f"({nb_tot} bins)")

        # Build grouped envelopes for reweighted
        rew_groups = _build_grouped_envelopes(
            'dphi', prior_data['d'], edges, t_cen, reweighted_dict,
            area_match_target=t_cen, var_map=var_map)

        # Build grouped envelopes for theory target
        tgt_groups = _build_target_grouped_envelopes('dphiDist', var_map)

        # ATLAS-style variable-width binning for dphi (fine at small d=π−Δφ
        # like ATLAS phi-star, coarser toward π). Metrics still on native bins.
        edges_orig = edges.copy()
        target_edges = np.unique(np.concatenate([
            np.arange(0.00, 0.20, 0.025),    # fine near peak (8 bins of 0.025)
            np.arange(0.20, 0.60, 0.05),     # 8 bins of 0.05
            np.arange(0.60, 1.20, 0.10),     # 6 bins of 0.10
            np.arange(1.20, 2.00, 0.20),     # 4 bins of 0.20
            np.array([2.00, 2.50, np.pi]),   # 2 wider bins toward π
        ]))
        target_edges = np.array([edges_orig[np.abs(edges_orig - e).argmin()]
                                 for e in target_edges])
        target_edges = np.unique(target_edges)
        rb_groups = []
        for k in range(len(target_edges) - 1):
            i0 = np.searchsorted(edges_orig, target_edges[k], side='left')
            i1 = np.searchsorted(edges_orig, target_edges[k+1], side='left')
            if i1 > i0:
                rb_groups.append((i0, i1))
        t_cen, edges = adaptive_rebin_density(t_cen, edges_orig, rb_groups)
        t_unc, _ = adaptive_rebin_density(t_unc, edges_orig, rb_groups)
        d_prior, _ = adaptive_rebin_density(d_prior, edges_orig, rb_groups)
        d_rew, _ = adaptive_rebin_density(d_rew, edges_orig, rb_groups)
        mask = t_cen > 0

        for grp in list(rew_groups.keys()):
            lo, hi, n = rew_groups[grp]
            lo, _ = adaptive_rebin_density(lo, edges_orig, rb_groups)
            hi, _ = adaptive_rebin_density(hi, edges_orig, rb_groups)
            rew_groups[grp] = (lo, hi, n)
        for grp in list(tgt_groups.keys()):
            lo, hi, n = tgt_groups[grp]
            lo, _ = adaptive_rebin_density(lo, edges_orig, rb_groups)
            hi, _ = adaptive_rebin_density(hi, edges_orig, rb_groups)
            tgt_groups[grp] = (lo, hi, n)

        # Total envelopes
        tgt_total_lo = t_cen.copy()
        tgt_total_hi = t_cen.copy()
        for grp, (lo, hi, n) in tgt_groups.items():
            tgt_total_lo = np.minimum(tgt_total_lo, lo)
            tgt_total_hi = np.maximum(tgt_total_hi, hi)
        rew_total_lo = d_rew.copy()
        rew_total_hi = d_rew.copy()
        for grp, (lo, hi, n) in rew_groups.items():
            rew_total_lo = np.minimum(rew_total_lo, lo)
            rew_total_hi = np.maximum(rew_total_hi, hi)

        # ── Multi-panel figure ──
        sub_groups = ['resum', 'fo', 'cs', 'kappa']
        sub_labels = {
            'resum': r'Resum.',
            'fo': r'FO',
            'cs': r'$C_0^{\mathrm{NP}}$',
            'kappa': r'$\kappa_{\mathrm{NP}}$',
        }
        n_sub = len(sub_groups)
        height_ratios = [3, 1.2] + [0.8] * n_sub
        fig, axes = plt.subplots(
            2 + n_sub, 1, figsize=(6.5, 10),
            height_ratios=height_ratios, sharex=True,
            gridspec_kw={'hspace': 0.0})
        ax = axes[0]
        axr = axes[1]

        # ── Upper panel ──
        _step_line(ax, edges, d_prior, color=C['prior'], lw=1.2, alpha=0.7,
                   label=r'Prior (Sherpa)', zorder=5)
        _step_line(ax, edges, d_rew, color=C['rew'], lw=1.6,
                   label=r'Reweighted', zorder=6)
        _step_line(ax, edges, t_cen, color=C['target'], lw=1.4,
                   label=r'Theory (central)', zorder=7)

        valid = t_cen[mask]
        if len(valid) > 0:
            ax.set_ylim(valid.min() * 0.3, valid.max() * 5)
        ax.set_yscale('log')
        ax.set_ylabel(r'$\mathrm{d}\sigma / \mathrm{d}(\pi - \Delta\phi)$')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper right', fontsize=9, ncol=1)

        acc_label = acc.replace("'", r"$'$").replace("+", r"$+$")
        ax.text(0.04, 0.96, r'$pp \to Z/\gamma^* \to \ell\ell$',
                transform=ax.transAxes, va='top', ha='left', fontsize=10)
        ax.text(0.04, 0.89, acc_label,
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                color='#555555')
        if td_prior_d > 0:
            ax.text(0.04, 0.82,
                    rf'$\chi^2/\mathrm{{bin}}$: {chi2b_prior_d:.1f} $\to$ {chi2b_rew_d:.1f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    color='#555555')

        # ── Main ratio ──
        with np.errstate(divide='ignore', invalid='ignore'):
            r_prior = np.where(mask, d_prior / t_cen, 1.0)
            r_rew = np.where(mask, d_rew / t_cen, 1.0)
            r_unc = np.where(mask, t_unc / t_cen, 0.0)
            r_tgt_lo = np.where(mask, tgt_total_lo / t_cen, 1.0)
            r_tgt_hi = np.where(mask, tgt_total_hi / t_cen, 1.0)
            r_rew_lo = np.where(mask, rew_total_lo / t_cen, 1.0)
            r_rew_hi = np.where(mask, rew_total_hi / t_cen, 1.0)

        _step_fill(axr, edges, r_tgt_lo, r_tgt_hi, color=C['target_scale'],
                   alpha=0.20, label=r'Theory unc.', zorder=1)
        _step_fill(axr, edges, 1 - r_unc, 1 + r_unc, color=C['target_stat'],
                   alpha=0.35, zorder=2)
        _step_fill(axr, edges, r_rew_lo, r_rew_hi, color=C['rew'],
                   alpha=0.15, label=r'Rew.\ unc.', zorder=3)
        axr.axhline(1.0, color='black', ls='-', lw=0.6, zorder=0)
        _step_line(axr, edges, r_prior, color=C['prior'], lw=1.2, alpha=0.7, zorder=5)
        _step_line(axr, edges, r_rew, color=C['rew'], lw=1.6, zorder=6)
        axr.set_ylabel(r'Ratio', fontsize=9)
        axr.set_ylim(0.85, 1.15)
        axr.xaxis.set_minor_locator(AutoMinorLocator())
        axr.yaxis.set_minor_locator(AutoMinorLocator())
        axr.legend(loc='upper right', fontsize=7, ncol=2)

        # ── Sub-ratio panels ──
        for si, grp in enumerate(sub_groups):
            axs = axes[2 + si]
            gc = VARIATION_GROUPS[grp]

            if grp in tgt_groups:
                lo, hi, nv = tgt_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask, lo / t_cen, 1.0)
                    rhi = np.where(mask, hi / t_cen, 1.0)
                _step_fill(axs, edges, rlo, rhi, color=gc['color'],
                           alpha=0.20, zorder=1)

            if grp in rew_groups:
                lo, hi, nv = rew_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask, lo / t_cen, 1.0)
                    rhi = np.where(mask, hi / t_cen, 1.0)
                _step_fill(axs, edges, rlo, rhi, color=gc['color'],
                           alpha=0.35, hatch='///', linewidth=0, zorder=2)

            axs.axhline(1.0, color='black', ls='-', lw=0.5, zorder=0)
            _step_line(axs, edges, r_rew, color=C['rew'], lw=1.0, alpha=0.5, zorder=3)
            axs.set_ylim(0.92, 1.08)
            axs.yaxis.set_minor_locator(AutoMinorLocator())
            axs.set_ylabel(sub_labels[grp], fontsize=8, rotation=0,
                           labelpad=20, va='center')
            if si < n_sub - 1:
                axs.tick_params(labelbottom=False)

        axes[-1].set_xlabel(r'$\pi - \Delta\phi$')
        axes[-1].xaxis.set_minor_locator(AutoMinorLocator())

        for ext in ['pdf', 'png']:
            p_ = f"{output_dir}/dphi_{acc_slug}.{ext}"
            fig.savefig(p_, dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_dir}/dphi_{acc_slug}.pdf (.png)")

    # ================================================================
    # PLOT: pT [GeV] (with theory target -- same multi-panel style as rT)
    # ================================================================
    if 'pTDist' in target and 'pT' in prior_data:
        edges = target['pTDist']['edges']
        t_cen = target['pTDist']['central']
        t_unc = target['pTDist']['central_unc']
        var_map = target['pTDist'].get('var_map', {})
        nbt = len(t_cen)

        d_prior = hist_to_density(prior_data['pT'], w_prior, edges)
        d_rew = hist_to_density(prior_data['pT'], w_rew, edges)
        mask = t_cen > 0
        d_prior = np.where(mask, d_prior, 0.0)
        d_rew = np.where(mask, d_rew, 0.0)
        d_prior = match_area(d_prior, t_cen, edges, label='pT prior')
        d_rew = match_area(d_rew, t_cen, edges, label='pT reweighted')

        td_prior_pT = triangle_divergence(d_prior, t_cen)
        td_rew_pT = triangle_divergence(d_rew, t_cen)
        chi2b_prior_pT, nbins_pT = chi2_per_bin(d_prior, t_cen, t_unc)
        chi2b_rew_pT, _ = chi2_per_bin(d_rew, t_cen, t_unc)
        print(f"  [TD] pT: Prior->Target = {td_prior_pT:.6e}, Rew->Target = {td_rew_pT:.6e}")
        print(f"  [χ²/bin] pT: Prior = {chi2b_prior_pT:.2f}, Rew = {chi2b_rew_pT:.2f}  ({nbins_pT} bins)")

        # Build grouped envelopes (reweighted + theory). Reweighted variation
        # densities live in reweighted_dict['pT'] on the SAME theory edges (the
        # run/plot mode binned them on theory edges when --pT_theory_file is set).
        rew_groups = _build_grouped_envelopes(
            'pT', prior_data['pT'], edges, t_cen, reweighted_dict,
            area_match_target=t_cen, var_map=var_map)
        # Drop any group whose length doesn't match theory bins (stale binning)
        rew_groups = {g: v for g, v in rew_groups.items() if len(v[0]) == nbt}
        tgt_groups = _build_target_grouped_envelopes('pTDist', var_map)

        # ATLAS-style variable-width binning (GeV): fine at the Sudakov peak,
        # coarser into the fixed-order tail. Theory native bins are 2 GeV.
        edges_orig = edges.copy()
        target_edges = np.unique(np.concatenate([
            np.arange(0.0,  20.0, 2.0),     # 2 GeV bins at the peak
            np.arange(20.0, 40.0, 4.0),     # 4 GeV bins
            np.arange(40.0, 80.0, 8.0),     # 8 GeV bins
            np.array([80.0, 100.0, 140.0, 200.0]),
        ]))
        target_edges = np.array([edges_orig[np.abs(edges_orig - e).argmin()]
                                 for e in target_edges])
        target_edges = np.unique(target_edges)
        rb_groups = []
        for k in range(len(target_edges) - 1):
            i0 = np.searchsorted(edges_orig, target_edges[k], side='left')
            i1 = np.searchsorted(edges_orig, target_edges[k+1], side='left')
            if i1 > i0:
                rb_groups.append((i0, i1))
        t_cen, edges = adaptive_rebin_density(t_cen, edges_orig, rb_groups)
        t_unc, _ = adaptive_rebin_density(t_unc, edges_orig, rb_groups)
        d_prior, _ = adaptive_rebin_density(d_prior, edges_orig, rb_groups)
        d_rew, _ = adaptive_rebin_density(d_rew, edges_orig, rb_groups)
        mask = t_cen > 0

        for grp in list(rew_groups.keys()):
            lo, hi, n = rew_groups[grp]
            lo, _ = adaptive_rebin_density(lo, edges_orig, rb_groups)
            hi, _ = adaptive_rebin_density(hi, edges_orig, rb_groups)
            rew_groups[grp] = (lo, hi, n)
        for grp in list(tgt_groups.keys()):
            lo, hi, n = tgt_groups[grp]
            lo, _ = adaptive_rebin_density(lo, edges_orig, rb_groups)
            hi, _ = adaptive_rebin_density(hi, edges_orig, rb_groups)
            tgt_groups[grp] = (lo, hi, n)

        # Total envelopes (already rebinned)
        tgt_total_lo = t_cen.copy()
        tgt_total_hi = t_cen.copy()
        for grp, (lo, hi, n) in tgt_groups.items():
            tgt_total_lo = np.minimum(tgt_total_lo, lo)
            tgt_total_hi = np.maximum(tgt_total_hi, hi)
        rew_total_lo = d_rew.copy()
        rew_total_hi = d_rew.copy()
        for grp, (lo, hi, n) in rew_groups.items():
            rew_total_lo = np.minimum(rew_total_lo, lo)
            rew_total_hi = np.maximum(rew_total_hi, hi)

        # ── Multi-panel figure ──
        sub_groups = ['resum', 'fo', 'cs', 'kappa']
        sub_labels = {
            'resum': r'Resum.',
            'fo': r'FO',
            'cs': r'$C_0^{\mathrm{NP}}$',
            'kappa': r'$\kappa_{\mathrm{NP}}$',
        }
        n_sub = len(sub_groups)
        height_ratios = [3, 1.2] + [0.8] * n_sub
        fig, axes = plt.subplots(
            2 + n_sub, 1, figsize=(6.5, 10),
            height_ratios=height_ratios, sharex=True,
            gridspec_kw={'hspace': 0.0})
        ax = axes[0]
        axr = axes[1]

        # Display window: full theory range (0–200 GeV)
        pT_disp_max = float(edges[-1])

        # ── Upper panel: clean, lines only ──
        _step_line(ax, edges, d_prior, color=C['prior'], lw=1.2, alpha=0.7,
                   label=r'Prior (Sherpa)', zorder=5)
        _step_line(ax, edges, d_rew, color=C['rew'], lw=1.6,
                   label=r'Reweighted', zorder=6)
        _step_line(ax, edges, t_cen, color=C['target'], lw=1.4,
                   label=r'Theory (central)', zorder=7)

        dmask = mask & (edges[:-1] < pT_disp_max)
        valid = t_cen[dmask]
        if len(valid) > 0:
            ax.set_ylim(valid.min() * 0.3, valid.max() * 5)
        ax.set_yscale('log')
        ax.set_xlim(edges[0], pT_disp_max)
        ax.set_ylabel(r'$\mathrm{d}\sigma / \mathrm{d}p_T^{\ell\ell}$ [GeV$^{-1}$]')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper right', fontsize=9, ncol=1)

        acc_label = acc.replace("'", r"$'$").replace("+", r"$+$")
        ax.text(0.04, 0.96, r'$pp \to Z/\gamma^* \to \ell\ell$',
                transform=ax.transAxes, va='top', ha='left', fontsize=10)
        ax.text(0.04, 0.89, acc_label,
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                color='#555555')
        ax.text(0.04, 0.82,
                rf'$\chi^2/\mathrm{{bin}}$: {chi2b_prior_pT:.1f} $\to$ {chi2b_rew_pT:.1f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                color='#555555')

        # ── Main ratio: total envelope ──
        with np.errstate(divide='ignore', invalid='ignore'):
            r_prior = np.where(mask, d_prior / t_cen, 1.0)
            r_rew = np.where(mask, d_rew / t_cen, 1.0)
            r_unc = np.where(mask, t_unc / t_cen, 0.0)
            r_tgt_lo = np.where(mask, tgt_total_lo / t_cen, 1.0)
            r_tgt_hi = np.where(mask, tgt_total_hi / t_cen, 1.0)
            r_rew_lo = np.where(mask, rew_total_lo / t_cen, 1.0)
            r_rew_hi = np.where(mask, rew_total_hi / t_cen, 1.0)

        _step_fill(axr, edges, r_tgt_lo, r_tgt_hi, color=C['target_scale'],
                   alpha=0.20, label=r'Theory unc.', zorder=1)
        _step_fill(axr, edges, 1 - r_unc, 1 + r_unc, color=C['target_stat'],
                   alpha=0.35, zorder=2)
        _step_fill(axr, edges, r_rew_lo, r_rew_hi, color=C['rew'],
                   alpha=0.15, label=r'Rew.\ unc.', zorder=3)
        axr.axhline(1.0, color='black', ls='-', lw=0.6, zorder=0)
        _step_line(axr, edges, r_prior, color=C['prior'], lw=1.2, alpha=0.7, zorder=5)
        _step_line(axr, edges, r_rew, color=C['rew'], lw=1.6, zorder=6)
        axr.set_ylabel(r'Ratio', fontsize=9)
        axr.set_ylim(0.85, 1.15)
        axr.xaxis.set_minor_locator(AutoMinorLocator())
        axr.yaxis.set_minor_locator(AutoMinorLocator())
        axr.legend(loc='upper right', fontsize=7, ncol=2)

        # ── Sub-ratio panels: one per uncertainty type ──
        for si, grp in enumerate(sub_groups):
            axs = axes[2 + si]
            gc = VARIATION_GROUPS[grp]

            if grp in tgt_groups:
                lo, hi, nv = tgt_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask, lo / t_cen, 1.0)
                    rhi = np.where(mask, hi / t_cen, 1.0)
                _step_fill(axs, edges, rlo, rhi, color=gc['color'],
                           alpha=0.20, zorder=1)

            if grp in rew_groups:
                lo, hi, nv = rew_groups[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask, lo / t_cen, 1.0)
                    rhi = np.where(mask, hi / t_cen, 1.0)
                _step_fill(axs, edges, rlo, rhi, color=gc['color'],
                           alpha=0.35, hatch='///', linewidth=0, zorder=2)

            axs.axhline(1.0, color='black', ls='-', lw=0.5, zorder=0)
            _step_line(axs, edges, r_rew, color=C['rew'], lw=1.0, alpha=0.5, zorder=3)
            axs.set_ylim(0.92, 1.08)
            axs.yaxis.set_minor_locator(AutoMinorLocator())
            axs.set_ylabel(sub_labels[grp], fontsize=8, rotation=0,
                           labelpad=20, va='center')
            if si < n_sub - 1:
                axs.tick_params(labelbottom=False)

        axes[-1].set_xlabel(r'$p_T^{\ell\ell}$ [GeV]')
        axes[-1].xaxis.set_minor_locator(AutoMinorLocator())
        axes[-1].set_xlim(edges[0], pT_disp_max)

        for ext in ['pdf', 'png']:
            p_ = f"{output_dir}/pT_{acc_slug}.{ext}"
            fig.savefig(p_, dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_dir}/pT_{acc_slug}.pdf (.png)")

    # ================================================================
    # PLOT: pT [GeV] (no theory target -- ratio is Rew/Prior)
    # ================================================================
    elif 'pT' in prior_data:
        pT_edges = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             12, 14, 16, 18, 20, 25, 30, 35, 40, 50,
                             60, 80, 100, 150, 200, 300, 500], dtype=float)
        pT_edges = pT_edges[pT_edges <= np.percentile(prior_data['pT'], 99.5)]
        if pT_edges[-1] < 200:
            pT_edges = np.append(pT_edges, 200)

        d_prior_pT = hist_to_density(prior_data['pT'], w_prior, pT_edges)
        d_rew_pT = hist_to_density(prior_data['pT'], w_rew, pT_edges)
        d_rew_pT = match_area(d_rew_pT, d_prior_pT, pT_edges, label='pT reweighted')

        # Build grouped envelopes for reweighted (no theory target)
        rew_groups_pT = _build_grouped_envelopes(
            'pT', prior_data['pT'], pT_edges, d_rew_pT, reweighted_dict)

        fig, ax, axr = _make_figure()

        # Plot grouped reweighted bands
        for zi, grp in enumerate(['resum', 'fo', 'cs', 'kappa']):
            if grp in rew_groups_pT:
                lo, hi, nv = rew_groups_pT[grp]
                gc = VARIATION_GROUPS[grp]
                _step_fill(ax, pT_edges, lo, hi, color=gc['color'], alpha=0.25,
                           label=rf'Rew.\ {gc["label"]}', zorder=3+zi,
                           hatch='///', linewidth=0)

        _step_line(ax, pT_edges, d_prior_pT, color=C['prior'], lw=1.4, alpha=0.85,
                   label=r'Prior (Sherpa)', zorder=11)
        _step_line(ax, pT_edges, d_rew_pT, color=C['rew'], lw=1.6,
                   label=r'Reweighted', zorder=12)

        mask_pT = d_prior_pT > 0
        valid = d_prior_pT[mask_pT]
        if len(valid) > 0:
            ax.set_ylim(valid.min() * 0.3, valid.max() * 5)
        ax.set_ylabel(r'$\mathrm{d}\sigma / \mathrm{d}p_T$ [GeV$^{-1}$]')

        with np.errstate(divide='ignore', invalid='ignore'):
            r_rew = np.where(mask_pT, d_rew_pT / d_prior_pT, 1.0)

        # Ratio panel: grouped reweighted bands
        for zi, grp in enumerate(['resum', 'fo', 'cs', 'kappa']):
            if grp in rew_groups_pT:
                lo, hi, nv = rew_groups_pT[grp]
                gc = VARIATION_GROUPS[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask_pT, lo / d_prior_pT, 1.0)
                    rhi = np.where(mask_pT, hi / d_prior_pT, 1.0)
                _step_fill(axr, pT_edges, rlo, rhi, color=gc['color'], alpha=0.25,
                           hatch='///', linewidth=0, zorder=3+zi)

        axr.axhline(1.0, color='black', ls='-', lw=0.6, zorder=0)
        _step_line(axr, pT_edges, r_rew, color=C['rew'], lw=1.6, zorder=12)

        _finish_axes(ax, axr, r'$p_T^{\ell\ell}$ [GeV]',
                     ratio_range=(0.7, 1.3), ratio_label=r'Rew.\ / Prior')
        _save(fig, output_dir, 'pT', acc_slug)

    # ================================================================
    # PLOT: mass [GeV] (no theory target)
    # ================================================================
    if 'm' in prior_data:
        mass_edges = np.linspace(50, 200, 31)

        d_prior_m = hist_to_density(prior_data['m'], w_prior, mass_edges)
        d_rew_m = hist_to_density(prior_data['m'], w_rew, mass_edges)
        d_rew_m = match_area(d_rew_m, d_prior_m, mass_edges, label='mass reweighted')

        # Build grouped envelopes for reweighted (no theory target)
        rew_groups_m = _build_grouped_envelopes(
            'mass', prior_data['m'], mass_edges, d_rew_m, reweighted_dict)

        fig, ax, axr = _make_figure()

        # Plot grouped reweighted bands
        for zi, grp in enumerate(['resum', 'fo', 'cs', 'kappa']):
            if grp in rew_groups_m:
                lo, hi, nv = rew_groups_m[grp]
                gc = VARIATION_GROUPS[grp]
                _step_fill(ax, mass_edges, lo, hi, color=gc['color'], alpha=0.25,
                           label=rf'Rew.\ {gc["label"]}', zorder=3+zi,
                           hatch='///', linewidth=0)

        _step_line(ax, mass_edges, d_prior_m, color=C['prior'], lw=1.4, alpha=0.85,
                   label=r'Prior (Sherpa)', zorder=11)
        _step_line(ax, mass_edges, d_rew_m, color=C['rew'], lw=1.6,
                   label=r'Reweighted', zorder=12)

        mask_m = d_prior_m > 0
        valid = d_prior_m[mask_m]
        if len(valid) > 0:
            ax.set_ylim(valid.min() * 0.3, valid.max() * 5)
        ax.set_ylabel(r'$\mathrm{d}\sigma / \mathrm{d}m_{\ell\ell}$ [GeV$^{-1}$]')

        with np.errstate(divide='ignore', invalid='ignore'):
            r_rew = np.where(mask_m, d_rew_m / d_prior_m, 1.0)

        # Ratio panel: grouped reweighted bands
        for zi, grp in enumerate(['resum', 'fo', 'cs', 'kappa']):
            if grp in rew_groups_m:
                lo, hi, nv = rew_groups_m[grp]
                gc = VARIATION_GROUPS[grp]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rlo = np.where(mask_m, lo / d_prior_m, 1.0)
                    rhi = np.where(mask_m, hi / d_prior_m, 1.0)
                _step_fill(axr, mass_edges, rlo, rhi, color=gc['color'], alpha=0.25,
                           hatch='///', linewidth=0, zorder=3+zi)

        axr.axhline(1.0, color='black', ls='-', lw=0.6, zorder=0)
        _step_line(axr, mass_edges, r_rew, color=C['rew'], lw=1.6, zorder=12)

        _finish_axes(ax, axr, r'$m_{\ell\ell}$ [GeV]',
                     ratio_range=(0.7, 1.3), ratio_label=r'Rew.\ / Prior')
        _save(fig, output_dir, 'mass', acc_slug)


def save_results(model, F_names, G_names, w_prior, w_rew, output_dir, acc, prior_data,
                 moments_csv=None, reweighted_dict=None, stat_band_weights=None,
                 rebin_factor=3, pT_theory_file=None):
    """Save lambdas and create plots"""
    acc_slug = acc.replace("'", "p").replace(" ", "")

    lam_path = f"{output_dir}/lambdas_{acc_slug}.csv"
    with open(lam_path, 'w') as f:
        f.write("i,j,moment_name,lambda,sigma,m_prior,target,pull\n")
        m_prior_arr = (model._ensure_prior_moments() if hasattr(model, '_ensure_prior_moments')
                       else model.m_prior)
        for p, (i, j) in enumerate(model.pairs):
            lam_val = model.lam[p]
            sig = model.sigma[p]
            m_p = m_prior_arr[p]
            tgt = model.targets[p]
            pull = (m_p - tgt) / sig if sig != 0 else 0.0
            name = _display_name(F_names[i], G_names[j])
            f.write(f"{i},{j},{name},{lam_val:.6e},{sig:.6e},{m_p:.6e},{tgt:.6e},{pull:.3f}\n")

    print(f"  Saved: {lam_path}")

    plot_distributions(prior_data, w_prior, w_rew, acc, output_dir, moments_csv,
                       reweighted_dict, stat_band_weights, rebin_factor=rebin_factor,
                       pT_theory_file=pT_theory_file)


# ========================================
# Helper: determine max_k needed from moment names
# ========================================
def max_k_from_moment_names(names):
    """Determine the maximum power of rT and dphi needed for a set of moment names."""
    max_rt = 0
    max_dphi = 0
    for name in names:
        parts = name.split('×')
        for part in parts:
            part = part.strip()
            m = re.match(r'(ln)?(rt|dphi)\^(\d+)', part)
            if m:
                var = m.group(2)
                k = int(m.group(3))
                if var == 'rt':
                    max_rt = max(max_rt, k)
                elif var == 'dphi':
                    max_dphi = max(max_dphi, k)
    return max_rt, max_dphi


def _find_hist_csvs(base_dir, acc_slug):
    """Find histogram CSV files for a given accuracy."""
    all_csvs = glob.glob(os.path.join(base_dir, "*.csv"))
    main_csvs = [f for f in all_csvs
                 if len(os.path.basename(f).split("__")) == 2 and f.endswith(".csv")]
    cand = [f for f in main_csvs if f"__{acc_slug}.csv" in os.path.basename(f)]
    return cand if cand else None


def _moment_matches_selection(o1, o2, name_set):
    """Check if a moment (o1, o2) matches any name in the normalized name set."""
    b1, k1 = parse_moment(o1)
    b2, k2 = parse_moment(o2)
    if b1 is None or b2 is None:
        return False
    if k1 == 0 and k2 == 0:
        return True
    candidates = []
    candidates.append(f"{b1}^{k1}×{b2}^{k2}")
    candidates.append(f"{b2}^{k2}×{b1}^{k1}")
    if k1 > 0 and k2 > 0:
        ckey = _composite_key(b1, k1, b2, k2)
        candidates.append(ckey.replace('*', '×'))
    if k1 == 0:
        candidates.append(f"const^0×{b2}^{k2}")
        candidates.append(f"{b2}^{k2}×const^0")
    if k2 == 0:
        candidates.append(f"{b1}^{k1}×const^0")
        candidates.append(f"const^0×{b1}^{k1}")
    return any(_normalize_moment_name(c) in name_set for c in candidates)


# ========================================
# Binned event surrogate (for fast selection)
# ========================================
def make_binned_prior(prior, rT_edges, dphi_edges, max_k=5, chunk=2_000_000):
    """Bin events into a 2D rT×dphi grid with EXACT per-cell moments.

    For each non-empty cell, stores the EXACT per-cell averages of:
      rT^k, lnrT^k for k=1..max_k    (rT-side basis moments)
      dphi^k, lndphi^k for k=1..max_k (dphi-side)
      cell weight = Σw, and weighted-mean pT, m for plotting.

    These exact per-cell moments avoid Jensen's inequality bias from using
    (mean rT)^k. Cell-product moments ⟨rT^a × dphi^b⟩ are still approximated
    as ⟨rT^a⟩_cell × ⟨dphi^b⟩_cell which is exact only to O(bin width²).

    Returned dict is consumed by build_features() which checks for 'rT_pow'/
    'lnrT_pow'/etc. keys and uses them instead of (rT/dphi).
    """
    rT = prior['rT']
    d  = prior['d']
    w  = prior['w']
    pT = prior.get('pT')
    m  = prior.get('m')

    n_rT = len(rT_edges) - 1
    n_d  = len(dphi_edges) - 1
    n_cells = n_rT * n_d

    # Accumulators (one per power per type)
    Wsum   = np.zeros(n_cells)
    rTk    = {k: np.zeros(n_cells) for k in range(1, max_k+1)}
    lnrTk  = {k: np.zeros(n_cells) for k in range(1, max_k+1)}
    dk     = {k: np.zeros(n_cells) for k in range(1, max_k+1)}
    lndk   = {k: np.zeros(n_cells) for k in range(1, max_k+1)}
    pTsum  = np.zeros(n_cells) if pT is not None else None
    msum   = np.zeros(n_cells) if m  is not None else None

    N = len(rT)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        rTc = rT[a:b]; dc = d[a:b]; wc = w[a:b]
        # Restrict to grid
        mk = (rTc >= rT_edges[0]) & (rTc < rT_edges[-1]) & \
             (dc  >= dphi_edges[0]) & (dc  < dphi_edges[-1]) & \
             np.isfinite(rTc) & np.isfinite(dc) & np.isfinite(wc)
        rTc, dc, wc = rTc[mk], dc[mk], wc[mk]
        if len(rTc) == 0:
            continue
        i_rT = np.searchsorted(rT_edges, rTc, side='right') - 1
        i_d  = np.searchsorted(dphi_edges, dc, side='right') - 1
        flat = i_rT * n_d + i_d
        Wsum += np.bincount(flat, weights=wc, minlength=n_cells)
        # Powers
        log_rT = np.log(np.maximum(rTc, 1e-30))
        log_d  = np.log(np.maximum(dc,  1e-30))
        rT_pow_k  = rTc.copy()
        lnrT_pow_k = log_rT.copy()
        d_pow_k   = dc.copy()
        lnd_pow_k = log_d.copy()
        for k in range(1, max_k+1):
            rTk[k]   += np.bincount(flat, weights=wc * rT_pow_k,   minlength=n_cells)
            lnrTk[k] += np.bincount(flat, weights=wc * lnrT_pow_k, minlength=n_cells)
            dk[k]    += np.bincount(flat, weights=wc * d_pow_k,    minlength=n_cells)
            lndk[k]  += np.bincount(flat, weights=wc * lnd_pow_k,  minlength=n_cells)
            if k < max_k:
                rT_pow_k *= rTc
                lnrT_pow_k *= log_rT
                d_pow_k *= dc
                lnd_pow_k *= log_d
        if pT is not None:
            pTsum += np.bincount(flat, weights=wc * pT[a:b][mk], minlength=n_cells)
        if m is not None:
            msum  += np.bincount(flat, weights=wc * m[a:b][mk],  minlength=n_cells)

    keep = Wsum > 0
    W = Wsum[keep]
    out = {'w': W,
           'rT': rTk[1][keep] / W,    # mean rT (for compatibility)
           'd':  dk[1][keep]  / W,    # mean d
           'rT_pow':   {k: rTk[k][keep]   / W for k in rTk},
           'lnrT_pow': {k: lnrTk[k][keep] / W for k in lnrTk},
           'dphi_pow': {k: dk[k][keep]    / W for k in dk},
           'lndphi_pow': {k: lndk[k][keep]/ W for k in lndk},
           'pT': (pTsum[keep] / W) if pT is not None else np.zeros_like(W),
           'm':  (msum[keep]  / W) if m  is not None else np.ones_like(W),
           '_binned': True,
           '_max_k': max_k,
           }
    print(f"  [Binned surrogate] {N:,} events → {keep.sum():,} non-empty "
          f"cells (of {n_cells} = {n_rT}×{n_d})")
    print(f"  Σw preserved: events={w.sum():.4e}, cells={W.sum():.4e}")
    return out


# ========================================
# Main
# ========================================
def main():
    import json
    args = get_args()

    print("="*80)
    print(f"MaxEnt Reweighting — Mode: {args.mode.upper()}")
    print("="*80)

    if getattr(args, 'gpu', False):
        print("  [GPU mode enabled — selection workers will use PyTorch backend]")

    prior = load_prior(args.prior_dir)
    print(f"Prior Z: {prior['w'].sum()}")

    # ── Cap events early + free RAM ──
    N_full = len(prior['rT'])
    if args.max_events is not None and args.max_events < N_full:
        N_full = args.max_events
        for key in prior:
            if hasattr(prior[key], '__len__') and len(prior[key]) > N_full:
                prior[key] = prior[key][:N_full].copy()
        import gc; gc.collect()
        print(f"  [Capped events to {N_full:,}]")

    for acc in args.accs:
        print("\n" + "="*80)
        print(f"Processing: {acc}")
        print("="*80)

        acc_slug = acc.replace("'", "p").replace(" ", "")
        mom_path = f"{args.mom_dir}/DYMoments_{acc_slug}.csv"
        if not os.path.exists(mom_path):
            print(f"  ERROR: {mom_path} not found")
            continue

        # Load moments with uncertainties
        moments = load_moments(mom_path)

        # Optionally swap the regularizer σ for the THEORY (scale-variation
        # envelope) uncertainty, floored at the statistical σ.
        if getattr(args, 'sigma_theory', False):
            sig_th = compute_sigma_theory(mom_path)
            n_swap = 0
            moments_st = []
            for o1, o2, val, unc in moments:
                st = sig_th.get((o1, o2), 0.0)
                base = unc if unc is not None else 0.0
                new_unc = max(st, base)
                if new_unc <= 0:
                    new_unc = unc
                if st > base:
                    n_swap += 1
                moments_st.append((o1, o2, val, new_unc))
            moments = moments_st
            print(f"  [σ_theory] regularizer σ ← scale-variation envelope "
                  f"(floored at σ_stat); {n_swap}/{len(moments)} moments "
                  f"theory-dominated")

        # Find histogram CSVs (try base_dir first, then mom_dir as fallback)
        hist_csv = args.hist_csv or _find_hist_csvs(args.base_dir, acc_slug) or _find_hist_csvs(args.mom_dir, acc_slug)

        # ════════════════════════════════════════════
        # SELECT MODE
        # ════════════════════════════════════════════
        if args.mode == "select":
            use_max_k_rt = args.select_max_k_rt
            use_max_k_dphi = args.select_max_k_dphi

            # Replace same-type marginal targets with values from distributions
            # (ensures consistency between moment targets and TD target)
            moments_filtered = list(moments)
            if args.dist_marginals and hist_csv is not None:
                composite = compute_composite_moments_from_distributions(
                    hist_csv, acc, max_k=max(use_max_k_rt, use_max_k_dphi))
                dist_lookup = {(o1, o2): (val, unc) for o1, o2, val, unc in composite}
                n_replaced = 0
                for i, (o1, o2, val, unc) in enumerate(moments_filtered):
                    b1, k1 = parse_moment(o1)
                    b2, k2 = parse_moment(o2)
                    if b1 is None or b2 is None:
                        continue
                    both_rt = _is_rt_type(b1) and _is_rt_type(b2) and k1 > 0 and k2 > 0
                    both_dphi = _is_dphi_type(b1) and _is_dphi_type(b2) and k1 > 0 and k2 > 0
                    if both_rt or both_dphi:
                        # Look up in distribution-computed values (try both orderings)
                        dval = dist_lookup.get((o1, o2)) or dist_lookup.get((o2, o1))
                        if dval is not None:
                            moments_filtered[i] = (o1, o2, dval[0], dval[1])
                            n_replaced += 1
                if n_replaced:
                    print(f"  Replaced {n_replaced} same-type marginal targets with distribution values")
            print(f"  [Select mode] {len(moments_filtered)} candidate moments, "
                  f"max_k_rt={use_max_k_rt}, max_k_dphi={use_max_k_dphi}")

            # Load target distributions for TD (also gives us bin edges for surrogate)
            print(f"\n  [Loading Target Distributions for TD]")
            target_dists = load_target_distributions(hist_csv, acc)
            if not target_dists:
                print("  ERROR: No target distributions found!")
                print("  Provide --hist_csv with binned rTDist/dphiDist data")
                continue
            coarse = getattr(args, 'coarse_bins', False)
            if getattr(args, 'rebin_atlas', False):
                # rT can be coarsened, but dphi stays at native (or full ATLAS).
                # Coarsening dphi reduces σ_target → tiny dphi χ² σ → blowup.
                rt_e = ATLAS_RT_EDGES_COARSE if coarse else ATLAS_RT_EDGES
                d_e = ATLAS_DPHI_EDGES  # always use the finer ATLAS dphi, not coarse
                print(f"  [Rebinning targets onto ATLAS-style{' (coarse rT)' if coarse else ''} edges]")
                if 'rTDist' in target_dists:
                    target_dists['rTDist'] = rebin_target_dist(target_dists['rTDist'], rt_e)
                    print(f"    rTDist: {len(rt_e)-1} bins (was 500)")
                if 'dphiDist' in target_dists:
                    target_dists['dphiDist'] = rebin_target_dist(target_dists['dphiDist'], d_e)
                    print(f"    dphiDist: {len(d_e)-1} bins (was 80)")
            elif getattr(args, 'rebin_log', False):
                rt_e = LOG_RT_EDGES_COARSE if coarse else LOG_RT_EDGES
                print(f"  [Rebinning rTDist onto log-uniform{' (coarse rT)' if coarse else ''} edges, dphi unchanged]")
                if 'rTDist' in target_dists:
                    target_dists['rTDist'] = rebin_target_dist(target_dists['rTDist'], rt_e)
                    print(f"    rTDist: {len(rt_e)-1} bins (was 500)")

            # Either binned surrogate, importance-sampled subsample, or random
            N_full = len(prior['w'])
            if args.binned_select:
                rT_edges = target_dists['rTDist']['edges']
                dphi_edges = target_dists['dphiDist']['edges']
                prior_select = make_binned_prior(prior, rT_edges, dphi_edges)
                n_select = len(prior_select['w'])
            elif args.is_select:
                n_select = min(args.select_n_events, N_full)
                print(f"  IS subsample: {N_full:,} → {n_select:,} events "
                      f"(probability ∝ |w_i|, seed=42)")
                rng = np.random.default_rng(42)
                abs_w = np.abs(prior['w'])
                p = abs_w / abs_w.sum()
                # Sample with replacement weighted by |w|
                idx_sub = rng.choice(N_full, size=n_select, replace=True, p=p)
                # Self-normalized IS: each sampled event represents ~ 1/p_i raw events,
                # so its effective weight in moments is sign(w_i) * mean(|w|).
                # Set new weight = sign(w_i) * (Σ|w| / N') so Σ_sample w_new ≈ Σ_full w.
                w_per = abs_w.sum() / n_select
                new_w = np.sign(prior['w'][idx_sub]) * w_per
                prior_select = {k: (v[idx_sub] if hasattr(v, '__len__') and len(v) == N_full
                                    else v) for k, v in prior.items()}
                prior_select['w'] = new_w.astype(np.float64)
                neff_full = (prior['w'].sum())**2 / np.sum(prior['w']**2)
                neff_sub = (new_w.sum())**2 / np.sum(new_w**2)
                print(f"  N_eff: full={neff_full:,.0f} ({100*neff_full/N_full:.1f}%), "
                      f"IS sub={neff_sub:,.0f} ({100*neff_sub/n_select:.1f}%)")
            else:
                n_select = min(args.select_n_events, N_full)
                if n_select < N_full:
                    print(f"  Subsampling prior: {N_full:,} → {n_select:,} events (random, seed=42)")
                    rng = np.random.default_rng(42)
                    idx_sub = rng.choice(N_full, size=n_select, replace=False)
                    idx_sub.sort()
                    prior_select = {k: v[idx_sub] if hasattr(v, '__len__') and len(v) == N_full
                                    else v for k, v in prior.items()}
                else:
                    prior_select = prior

            F, G, F_names, G_names = build_features(
                prior_select, moments_filtered, use_max_k_rt, use_max_k_dphi,
                winsorize_pct=args.winsorize_pct)

            pairs, targets, sigmas = extract_pairs(
                F_names, G_names, moments_filtered, use_max_k_rt, use_max_k_dphi)

            # Remove normalization constraint, apply max_pair_power filter
            nontrivial = []
            for idx in range(len(pairs)):
                i, j = pairs[idx]
                name = _display_name(F_names[i], G_names[j])
                if name == 'const^0×const^0':
                    continue
                pw = _pair_total_power(pairs[idx], F_names, G_names)
                if pw > args.max_pair_power:
                    continue
                nontrivial.append(idx)

            all_pairs = pairs[nontrivial]
            all_targets = targets[nontrivial]
            all_sigmas = [sigmas[k] for k in nontrivial]

            print(f"  Candidate moments for selection: {len(all_pairs)} "
                  f"(max_k={use_max_k_rt}/{use_max_k_dphi}, max_pair_power={args.max_pair_power})")
            for idx in range(len(all_pairs)):
                i, j = all_pairs[idx]
                name = _display_name(F_names[i], G_names[j])
                print(f"    {idx+1:3d}. {name}")

            # Run greedy selection
            if args.regularize:
                sigmas_for_select = all_sigmas
                print(f"  Selection with target σ regularization")
            else:
                sigmas_for_select = [0.0] * len(all_sigmas)
                print(f"  Selection with σ=0 (pure moment matching)")

            # Resolve --preselect names to indices in all_pairs
            preselected_indices = []
            if args.preselect:
                preselect_names = set(
                    _normalize_moment_name(n.strip())
                    for n in args.preselect.split(','))
                for pidx in range(len(all_pairs)):
                    i_, j_ = all_pairs[pidx]
                    nm = _display_name(F_names[i_], G_names[j_])
                    if _normalize_moment_name(nm) in preselect_names:
                        preselected_indices.append(pidx)
                found_names = {_normalize_moment_name(
                    _display_name(F_names[all_pairs[p][0]], G_names[all_pairs[p][1]]))
                    for p in preselected_indices}
                missing = preselect_names - found_names
                if missing:
                    print(f"  WARNING: --preselect moments not found in candidate list: {missing}")

            selected_idx, selection_log = greedy_select_by_td(
                F, G, all_pairs, all_targets, sigmas_for_select, prior_select['w'],
                prior_select, target_dists, F_names, G_names,
                max_moments=args.select_max_moments,
                n_events=n_select,
                newton_steps=args.max_newton_steps,
                newton_tol=args.newton_tol,
                screen_top_k=args.screen_top_k,
                screen_steps=args.screen_steps,
                n_workers=args.n_workers,
                min_improvement_pct=args.select_min_improvement,
                min_steps=args.select_min_steps,
                max_component_worsen=args.max_component_worsen,
                preselected_indices=preselected_indices or None,
                no_backward=args.no_backward,
                select_strategy=args.select_strategy,
                use_gpu=getattr(args, 'gpu', False),
                phase1_pure_rt=getattr(args, 'phase1_pure_rt', False),
            )

            # ── Save JSON with selected moments ──
            # Use the actual selected indices (accounts for backward removals)
            selected_names = []
            for idx in selected_idx:
                i_, j_ = all_pairs[idx]
                selected_names.append(_display_name(F_names[i_], G_names[j_]))

            json_path = f"{args.output_dir}/selected_moments_{acc_slug}.json"
            # Make selection_log JSON-serializable (convert numpy)
            log_for_json = []
            for e in selection_log:
                entry = {}
                for k, v in e.items():
                    if isinstance(v, (np.integer,)):
                        entry[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        entry[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        entry[k] = v.tolist()
                    else:
                        entry[k] = v
                log_for_json.append(entry)

            td_baseline_val = (selection_log[0]['td'] + selection_log[0]['delta_td']
                               if selection_log else 0)
            result = {
                'accuracy': acc,
                'selected_moments': selected_names,
                'selection_log': log_for_json,
                'prior_td': {'total': float(td_baseline_val)},
                'final_td': float(selection_log[-1]['td']) if selection_log else 0,
                'n_events_used': min(args.select_n_events, N_full),
            }
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n  ✓ Saved selected moments: {json_path}")
            print(f"    {len(selected_names)} moments selected")
            print(f"    Use with: --mode run --moments_file {json_path}")

            # Save CSV log too
            log_path = f"{args.output_dir}/moment_selection_{acc_slug}.csv"
            with open(log_path, 'w') as f:
                f.write("step,moments,n_moments,td,td_rT,td_dphi,delta_td,"
                        "pct_improvement,rms_pull,neff\n")
                for e in selection_log:
                    names_str = '|'.join(e['names']) if 'names' in e else e['name']
                    n_mom = len(e['names']) if 'names' in e else 1
                    f.write(f"{e['step']},{names_str},{n_mom},"
                            f"{e['td']:.6f},{e['td_rT']:.6f},{e['td_dphi']:.6f},"
                            f"{e['delta_td']:.6f},{e['pct_improvement']:.2f},"
                            f"{e['rms_pull']:.6f},{e['neff']:.6f}\n")
            print(f"  ✓ Saved selection log: {log_path}")

            # Quick final fit on current dataset for preview
            if selected_idx:
                print(f"\n{'='*80}")
                print(f"[Preview Fit: {len(selected_idx)} moments on {n_select:,} events]")
                print(f"{'='*80}")

                sel_pairs = all_pairs[selected_idx]
                sel_targets = all_targets[selected_idx]
                sel_sigmas_final = [all_sigmas[k] for k in selected_idx] \
                    if args.regularize else [0.0] * len(selected_idx)

                model_final = MaxEntDual(
                    F, G, sel_pairs, sel_targets, prior_select['w'],
                    sigmas_target=sel_sigmas_final,
                    F_names=F_names, G_names=G_names)
                optimize_newton(model_final, max_steps=args.max_newton_steps,
                                tol=args.newton_tol, verbose=True)

                print_diagnostics(model_final, F_names, G_names,
                                  top_k=min(50, len(selected_idx)))

                w_rew = model_final.get_weights()
                n_eff = (np.sum(w_rew)**2 / np.sum(w_rew**2))
                print(f"\n  N_eff = {n_eff:.1f} ({100*n_eff/len(w_rew):.2f}%)")

                td_rT_f, td_d_f, chi2pb_rT_f, chi2pb_d_f = compute_td_split(
                    prior_select, w_rew, target_dists, len(w_rew))
                w_prior_norm = prior_select['w'] / prior_select['w'].sum()
                td_rT_p, td_d_p, chi2pb_rT_p, chi2pb_d_p = compute_td_split(
                    prior_select, w_prior_norm, target_dists, len(w_rew))
                print(f"\n  [TD on {n_select:,} events]")
                print(f"    Prior:      TD={td_rT_p+td_d_p:.4f}  "
                      f"(rT={td_rT_p:.4f}, dphi={td_d_p:.4f})")
                print(f"    Reweighted: TD={td_rT_f+td_d_f:.4f}  "
                      f"(rT={td_rT_f:.4f}, dphi={td_d_f:.4f})")
                pct_imp = 100 * (1 - (td_rT_f+td_d_f) / (td_rT_p+td_d_p))
                print(f"    Improvement: {pct_imp:.1f}%")
                print(f"\n  [χ²/bin on {n_select:,} events]")
                print(f"    Prior:      rT={chi2pb_rT_p:.2f}, dphi={chi2pb_d_p:.2f}")
                print(f"    Reweighted: rT={chi2pb_rT_f:.2f}, dphi={chi2pb_d_f:.2f}")
                if chi2pb_d_f > chi2pb_d_p or chi2pb_rT_f > chi2pb_rT_p:
                    print(f"    WARNING: reweighting worsened χ²/bin in at least one observable."
                          f" Review selected moments.")

                save_results(model_final, F_names, G_names, prior_select['w'],
                             w_rew, args.output_dir, acc, prior_select, hist_csv,
                             rebin_factor=args.rebin_factor)

        # ════════════════════════════════════════════
        # RUN MODE
        # ════════════════════════════════════════════
        elif args.mode == "run":
            # Load selected moments from JSON or fallback to SELECTED_MOMENT_NAMES
            if args.moments_file:
                with open(args.moments_file, 'r') as f:
                    sel_data = json.load(f)
                selected_names = set(sel_data['selected_moments'])
                print(f"  Loaded {len(selected_names)} moments from {args.moments_file}")
            elif SELECTED_MOMENT_NAMES is not None:
                selected_names = SELECTED_MOMENT_NAMES
                print(f"  Using {len(selected_names)} moments from SELECTED_MOMENT_NAMES")
            else:
                selected_names = None
                print(f"  No moment selection — using all available moments")

            # Auto-derive max_k from selected moment names
            if selected_names is not None:
                use_max_k_rt, use_max_k_dphi = max_k_from_moment_names(selected_names)
                # Also check for composite lnrt/lndphi terms
                for name in selected_names:
                    parts = name.split('×')
                    for part in parts:
                        m = re.match(r'ln(rt|dphi)\^(\d+)', part.strip())
                        if m:
                            var = m.group(1)
                            k = int(m.group(2))
                            if var == 'rt':
                                use_max_k_rt = max(use_max_k_rt, k)
                            else:
                                use_max_k_dphi = max(use_max_k_dphi, k)
                # Ensure at least 1
                use_max_k_rt = max(use_max_k_rt, 1)
                use_max_k_dphi = max(use_max_k_dphi, 1)
                print(f"  Auto-derived max_k: rT={use_max_k_rt}, dphi={use_max_k_dphi}")
            else:
                use_max_k_rt = 3
                use_max_k_dphi = 3

            # varmT CSV already includes all marginal self-product moments
            # with proper theory uncertainties — no need to compute from distributions

            # Filter moments
            if selected_names is not None:
                normalized_selected = {_normalize_moment_name(n)
                                       for n in selected_names}
                moments_filtered = [m for m in moments
                                    if _moment_matches_selection(
                                        m[0], m[1], normalized_selected)]
                print(f"  Filtered: {len(moments_filtered)} / {len(moments)} "
                      f"moments match selection")
            else:
                normalized_selected = None
                moments_filtered = moments

            F, G, F_names, G_names = build_features(
                prior, moments_filtered, use_max_k_rt, use_max_k_dphi,
                winsorize_pct=args.winsorize_pct)

            pairs, targets, sigmas = extract_pairs(
                F_names, G_names, moments_filtered,
                use_max_k_rt, use_max_k_dphi)

            # Subselect by name
            all_pairs = pairs
            all_targets = targets

            sel_pairs, sel_targets, sel_sigmas = [], [], []
            for idx, (i, j) in enumerate(all_pairs):
                name = _display_name(F_names[i], G_names[j])
                if (normalized_selected is None
                        or _normalize_moment_name(name) in normalized_selected):
                    sel_pairs.append((i, j))
                    sel_targets.append(all_targets[idx])
                    sel_sigmas.append(sigmas[idx])

            if len(sel_pairs) == 0:
                print("  ERROR: No constraints matched selection!")
                continue

            print(f"  Using {len(sel_pairs)} / {len(all_pairs)} moments")

            # Run mode regularization
            if args.regularize:
                print(f"  Using target σ as regularization")
                print(f"  σ range: [{min(sel_sigmas):.3e}, {max(sel_sigmas):.3e}]")
            else:
                sel_sigmas = [0.0] * len(sel_pairs)
                print(f"  No regularization (run mode): σ = 0 for all moments")

            pairs = np.array(sel_pairs, dtype=np.int64)
            targets_arr = np.array(sel_targets, dtype=np.float64)

            # Event-staged optimization
            event_stages = args.event_stages
            event_counts = []
            for es in range(event_stages - 1, -1, -1):
                n_ev = max(N_full // (2 ** es), 10000)
                event_counts.append(n_ev)
            event_counts = sorted(set(event_counts))

            print(f"\n[Event Staging: {len(event_counts)} stages: "
                  f"{', '.join(f'{n:,}' for n in event_counts)}]")

            lam = np.zeros(len(pairs), dtype=np.float64)

            for ev_stage, n_events in enumerate(event_counts):
                F_sub = F[:n_events]
                G_sub = G[:n_events]
                w_sub = prior['w'][:n_events]

                is_final = (n_events == N_full)
                label = f"{n_events:,}" + (" (full)" if is_final else "")

                print(f"\n{'='*80}")
                print(f"[Event Stage {ev_stage+1}/{len(event_counts)}: "
                      f"{label} events]")
                print(f"{'='*80}")

                model = MaxEntDual(F_sub, G_sub, pairs, targets_arr, w_sub,
                                   sigmas_target=sel_sigmas,
                                   F_names=F_names, G_names=G_names)
                model.lam = lam.copy()

                optimize_newton(model, max_steps=args.max_newton_steps,
                                tol=args.newton_tol, verbose=args.verbose)

                dlam = np.abs(model.lam - lam)
                print(f"\n  [λ Change] max|Δλ|={dlam.max():.4e}, "
                      f"mean|Δλ|={dlam.mean():.4e}")

                lam = model.lam.copy()

            # Final model on full dataset
            print(f"\n{'='*80}")
            print("[Final Evaluation on Full Dataset]")
            print(f"{'='*80}")

            model_final = MaxEntDual(F, G, pairs, targets_arr, prior['w'],
                                      sigmas_target=sel_sigmas,
                                      F_names=F_names, G_names=G_names)
            model_final.lam = lam.copy()

            print_diagnostics(model_final, F_names, G_names,
                              top_k=min(50, len(pairs)))

            w_rew = model_final.get_weights()

            n_eff = (np.sum(w_rew)**2 / np.sum(w_rew**2))
            print(f"\n  N_eff = {n_eff:.1f} ({100*n_eff/len(w_rew):.2f}%)")

            # Save per-event reweighted weights
            weights_path = f"{args.output_dir}/weights_{acc_slug}.csv.gz"
            pd.DataFrame({"w_rew": w_rew}).to_csv(
                weights_path, index=False, compression="gzip")
            print(f"  Saved per-event weights: {weights_path} "
                  f"({len(w_rew):,} events)")

            # ── Stat uncertainty band: fit target ± σ_stat ──
            stat_band_weights = None
            if args.stat_band:
                print(f"\n[Stat Uncertainty Band: fitting target ± σ_stat]")

                # Need the original (non-uniform) sigmas for the ± shift
                sel_sigmas_orig = []
                for idx, (i, j) in enumerate(all_pairs):
                    name = _display_name(F_names[i], G_names[j])
                    if (normalized_selected is None
                            or _normalize_moment_name(name)
                            in normalized_selected):
                        sel_sigmas_orig.append(sigmas[idx])

                targets_up = targets_arr + np.array(sel_sigmas_orig)
                targets_down = targets_arr - np.array(sel_sigmas_orig)

                for label, tgt_shifted in [("upper (+σ)", targets_up),
                                           ("lower (-σ)", targets_down)]:
                    model_sb = MaxEntDual(
                        F, G, pairs, tgt_shifted, prior['w'],
                        sigmas_target=sel_sigmas,
                        F_names=F_names, G_names=G_names)
                    model_sb.lam = lam.copy()  # warm-start from central
                    optimize_newton(model_sb,
                                    max_steps=args.max_newton_steps,
                                    tol=args.newton_tol, verbose=False)
                    w_sb = model_sb.get_weights()
                    n_eff_sb = (np.sum(w_sb)**2 / np.sum(w_sb**2))
                    print(f"  {label}: N_eff={n_eff_sb:.0f} "
                          f"({100*n_eff_sb/len(w_sb):.2f}%)")

                    if stat_band_weights is None:
                        stat_band_weights = (w_sb, None)
                    else:
                        stat_band_weights = (stat_band_weights[0], w_sb)

            # Reweight variations
            reweighted_dict = None
            if args.reweight_variations:
                import gc
                print(f"\n[Reweighting All Scale Variations]")
                central_scale, var_scales = get_scale_variations(mom_path)

                # Load target distributions for histogram edges
                target_dists = load_target_distributions(hist_csv, acc)

                # Accumulate lambdas for reproducibility
                all_lambdas = {}

                # Pre-compute histogram edges for all observables
                hist_edges = {}
                hist_edges_data = {}  # edges-only for pickling to workers
                if 'rTDist' in target_dists:
                    hist_edges['rT'] = (prior['rT'], target_dists['rTDist']['edges'])
                    hist_edges_data['rT'] = target_dists['rTDist']['edges']
                if 'dphiDist' in target_dists:
                    hist_edges['dphi'] = (prior['d'], target_dists['dphiDist']['edges'])
                    hist_edges_data['dphi'] = target_dists['dphiDist']['edges']
                if 'pT' in prior and 'm' in prior:
                    pT_edges = None
                    # If a pT theory file is given, bin variations on theory edges
                    # so reweighted envelopes overlay the theory in the fancy plot.
                    if args.pT_theory_file:
                        _pt_t = load_pT_theory(args.pT_theory_file, acc)
                        if _pt_t is not None:
                            pT_edges = _pt_t['edges']
                            print(f"  pT binning from theory: {len(pT_edges)-1} bins "
                                  f"[{pT_edges[0]:.0f},{pT_edges[-1]:.0f}] GeV")
                    if pT_edges is None:
                        pT_edges = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                             12, 14, 16, 18, 20, 25, 30, 35, 40, 50,
                                             60, 80, 100, 150, 200, 300, 500], dtype=float)
                        pT_edges = pT_edges[pT_edges <= np.percentile(prior['pT'], 99.5)]
                        if pT_edges[-1] < 200:
                            pT_edges = np.append(pT_edges, 200)
                    hist_edges['pT'] = (prior['pT'], pT_edges)
                    hist_edges_data['pT'] = pT_edges
                    hist_edges['mass'] = (
                        prior['m'], np.linspace(50, 200, 31))
                    hist_edges_data['mass'] = np.linspace(50, 200, 31)

                # Pre-histogram central weights
                reweighted_dict = {
                    central_scale: precompute_variation_hists(w_rew, hist_edges)
                }
                all_lambdas[central_scale] = lam.copy()

                # Pre-compute targets for all variations (fast, serial)
                var_tasks = []
                for scale_fo, scale_res in var_scales:
                    moments_var = load_moments_for_scale(
                        mom_path, scale_fo, scale_res)
                    if moments_var is None:
                        continue

                    pairs_var, targets_var, sigmas_var = extract_pairs(
                        F_names, G_names, moments_var,
                        use_max_k_rt, use_max_k_dphi)

                    sel_p_v, sel_t_v, sel_s_v = [], [], []
                    for idx, (i, j) in enumerate(pairs_var):
                        name = _display_name(F_names[i], G_names[j])
                        if (normalized_selected is None
                                or _normalize_moment_name(name)
                                in normalized_selected):
                            sel_p_v.append((i, j))
                            sel_t_v.append(targets_var[idx])
                            sel_s_v.append(sigmas_var[idx])

                    if len(sel_p_v) == 0:
                        continue

                    pairs_arr = np.array(sel_p_v, dtype=np.int64)
                    targets_arr_v = np.array(sel_t_v, dtype=np.float64)
                    sel_s_v_orig = list(sel_s_v)

                    if not args.regularize:
                        sel_s_v = [0.0] * len(sel_s_v)

                    var_tasks.append((
                        scale_fo, scale_res,
                        pairs_arr, targets_arr_v, sel_s_v,
                        lam.copy(),  # warm-start from central
                        args.stat_band, sel_s_v_orig
                    ))

                print(f"  Prepared {len(var_tasks)} variation tasks")

                if args.n_workers > 1 and len(var_tasks) > 1:
                    import multiprocessing as mp
                    import platform
                    if platform.system() == 'Darwin':
                        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
                    ctx = mp.get_context('fork')

                    # Initialize shared data for variation workers
                    _mp_var_init(F, G, prior['w'], F_names, G_names,
                                 hist_edges_data,
                                 prior['rT'], prior['d'],
                                 prior.get('pT'), prior.get('m'),
                                 args.max_newton_steps, args.newton_tol)

                    n_var_workers = min(args.n_workers, len(var_tasks))
                    print(f"  Running with {n_var_workers} workers (fork)")

                    pool = ctx.Pool(
                        processes=n_var_workers,
                        initializer=_mp_var_init,
                        initargs=(F, G, prior['w'], F_names, G_names,
                                  hist_edges_data,
                                  prior['rT'], prior['d'],
                                  prior.get('pT'), prior.get('m'),
                                  args.max_newton_steps, args.newton_tol))

                    for count, result_list in enumerate(
                            pool.imap_unordered(_mp_reweight_variation, var_tasks)):
                        for key, hists, lam_v, rms_v, neff_v in result_list:
                            reweighted_dict[key] = hists
                            all_lambdas[key] = lam_v
                            if isinstance(key, tuple) and len(key) == 2:
                                print(f"  {count+1}/{len(var_tasks)}: "
                                      f"{key[0]}/{key[1]}  "
                                      f"RMS pull={rms_v:.4f}, N_eff={neff_v:.1f}%")

                    pool.close()
                    pool.join()
                else:
                    # Serial fallback
                    _mp_var_init(F, G, prior['w'], F_names, G_names,
                                 hist_edges_data,
                                 prior['rT'], prior['d'],
                                 prior.get('pT'), prior.get('m'),
                                 args.max_newton_steps, args.newton_tol)

                    for count, task in enumerate(var_tasks):
                        result_list = _mp_reweight_variation(task)
                        for key, hists, lam_v, rms_v, neff_v in result_list:
                            reweighted_dict[key] = hists
                            all_lambdas[key] = lam_v
                            if isinstance(key, tuple) and len(key) == 2:
                                print(f"  {count+1}/{len(var_tasks)}: "
                                      f"{key[0]}/{key[1]}  "
                                      f"RMS pull={rms_v:.4f}, N_eff={neff_v:.1f}%")

                print(f"\n  Reweighted {len(reweighted_dict)} variations "
                      f"(incl. stat bands)" if args.stat_band
                      else f"\n  Reweighted {len(reweighted_dict)} variations")

                # Save all lambdas for reproducibility
                import json
                lam_all_path = f"{args.output_dir}/lambdas_all_variations_{acc_slug}.json"
                lam_save = {}
                for key, lam_arr in all_lambdas.items():
                    key_str = '/'.join(str(k) for k in key) if isinstance(key, tuple) else str(key)
                    lam_save[key_str] = lam_arr.tolist()
                with open(lam_all_path, 'w') as f:
                    json.dump(lam_save, f, indent=2)
                print(f"  Saved all variation lambdas: {lam_all_path} "
                      f"({len(all_lambdas)} sets)")

            save_results(model_final, F_names, G_names, prior['w'], w_rew,
                         args.output_dir, acc, prior, hist_csv,
                         reweighted_dict, stat_band_weights,
                         rebin_factor=args.rebin_factor,
                         pT_theory_file=args.pT_theory_file)

        # ════════════════════════════════════════════
        # PLOT MODE — replot from saved lambdas
        # ════════════════════════════════════════════
        elif args.mode == "plot":
            # Load selected moments (same logic as run mode)
            if args.moments_file:
                with open(args.moments_file, 'r') as f:
                    sel_data = json.load(f)
                selected_names = set(sel_data['selected_moments'])
                print(f"  Loaded {len(selected_names)} moments from {args.moments_file}")
            elif SELECTED_MOMENT_NAMES is not None:
                selected_names = SELECTED_MOMENT_NAMES
            else:
                selected_names = None

            # Auto-derive max_k
            if selected_names is not None:
                use_max_k_rt, use_max_k_dphi = max_k_from_moment_names(selected_names)
                for name in selected_names:
                    parts = name.split('×')
                    for part in parts:
                        m = re.match(r'ln(rt|dphi)\^(\d+)', part.strip())
                        if m:
                            var = m.group(1)
                            k = int(m.group(2))
                            if var == 'rt':
                                use_max_k_rt = max(use_max_k_rt, k)
                            else:
                                use_max_k_dphi = max(use_max_k_dphi, k)
                use_max_k_rt = max(use_max_k_rt, 1)
                use_max_k_dphi = max(use_max_k_dphi, 1)
                print(f"  Auto-derived max_k: rT={use_max_k_rt}, dphi={use_max_k_dphi}")
            else:
                use_max_k_rt, use_max_k_dphi = 3, 3

            # Filter moments and build features
            if selected_names is not None:
                normalized_selected = {_normalize_moment_name(n)
                                       for n in selected_names}
                moments_filtered = [m_ for m_ in moments
                                    if _moment_matches_selection(
                                        m_[0], m_[1], normalized_selected)]
            else:
                normalized_selected = None
                moments_filtered = moments

            F, G, F_names, G_names = build_features(
                prior, moments_filtered, use_max_k_rt, use_max_k_dphi,
                winsorize_pct=args.winsorize_pct)

            pairs, targets, sigmas = extract_pairs(
                F_names, G_names, moments_filtered,
                use_max_k_rt, use_max_k_dphi)

            # Subselect by name
            sel_pairs, sel_targets, sel_sigmas = [], [], []
            for idx, (i, j) in enumerate(pairs):
                name = _display_name(F_names[i], G_names[j])
                if (normalized_selected is None
                        or _normalize_moment_name(name) in normalized_selected):
                    sel_pairs.append((i, j))
                    sel_targets.append(targets[idx])
                    sel_sigmas.append(sigmas[idx])

            pairs_arr = np.array(sel_pairs, dtype=np.int64)
            targets_arr = np.array(sel_targets, dtype=np.float64)

            # Find lambdas JSON
            lam_json = args.lambdas_json
            if lam_json is None:
                lam_json = f"{args.output_dir}/lambdas_all_variations_{acc_slug}.json"
            if not os.path.exists(lam_json):
                print(f"  ERROR: Lambdas file not found: {lam_json}")
                continue

            with open(lam_json, 'r') as f:
                all_lam = json.load(f)
            print(f"  Loaded {len(all_lam)} variation lambdas from {lam_json}")

            # Build histogram edges
            target_dists = load_target_distributions(hist_csv, acc)
            hist_edges = {}
            if 'rTDist' in target_dists:
                hist_edges['rT'] = (prior['rT'], target_dists['rTDist']['edges'])
            if 'dphiDist' in target_dists:
                hist_edges['dphi'] = (prior['d'], target_dists['dphiDist']['edges'])
            if 'pT' in prior and 'm' in prior:
                pT_edges = None
                if args.pT_theory_file:
                    _pt_t = load_pT_theory(args.pT_theory_file, acc)
                    if _pt_t is not None:
                        pT_edges = _pt_t['edges']
                if pT_edges is None:
                    pT_edges = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                         12, 14, 16, 18, 20, 25, 30, 35, 40, 50,
                                         60, 80, 100, 150, 200, 300, 500], dtype=float)
                    pT_edges = pT_edges[pT_edges <= np.percentile(prior['pT'], 99.5)]
                    if pT_edges[-1] < 200:
                        pT_edges = np.append(pT_edges, 200)
                hist_edges['pT'] = (prior['pT'], pT_edges)
                hist_edges['mass'] = (prior['m'], np.linspace(50, 200, 31))

            # Reconstruct weights and histograms for each variation
            # Build a lightweight model just for get_weights
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                model_base = MaxEntDual(F, G, pairs_arr, targets_arr,
                                        prior['w'],
                                        sigmas_target=sel_sigmas if args.regularize else None,
                                        F_names=F_names, G_names=G_names)

            reweighted_dict = {}
            central_key = None
            for key_str, lam_list in all_lam.items():
                lam_v = np.array(lam_list, dtype=np.float64)
                if len(lam_v) != len(model_base.lam):
                    print(f"  SKIP {key_str}: lambda length mismatch "
                          f"({len(lam_v)} vs {len(model_base.lam)})")
                    continue

                w_v = model_base.get_weights(lam_v)
                hists = precompute_variation_hists(w_v, hist_edges)

                # Parse key back to tuple
                parts = key_str.split('/')
                if len(parts) == 2:
                    key_tuple = (parts[0], parts[1])
                else:
                    key_tuple = key_str

                reweighted_dict[key_tuple] = hists

                # Identify central
                if 'CV->FO' in key_str and 'CV->Res' in key_str:
                    central_key = key_tuple
                    w_rew = w_v

                del w_v

            print(f"  Reconstructed {len(reweighted_dict)} variation histograms")

            n_eff = (np.sum(w_rew)**2 / np.sum(w_rew**2))
            print(f"  N_eff = {n_eff:.1f} ({100*n_eff/len(w_rew):.2f}%)")

            # Plot
            plot_distributions(prior, prior['w'], w_rew, acc,
                               args.output_dir, hist_csv,
                               reweighted_dict, None,
                               rebin_factor=args.rebin_factor,
                               pT_theory_file=args.pT_theory_file)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()