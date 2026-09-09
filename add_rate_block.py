"""Add the total-rate normalization ("rate" block) to the delivered lambda JSONs.

The MaxEnt weight w = w0 * exp(sum_k lambda_k phi_k - C) preserves the prior's
normalization (sum w = sum w0): only the SHAPE of the calculation is transferred.
The calculation's total rate sigma_calc (its 'dphi^0 x lndphi^0' moment, central and
per scheme) is therefore applied as an overall factor

    K = sigma_calc / sigma_prior            (central)
    K_s = sigma_calc,s / sigma_prior        (scheme s)

which leaves every normalized moment unchanged and so commutes with the fit.
sigma_prior is the mean stored event weight of the prior sample (POWHEG: pb per
event); --sigma-prior overrides it (e.g. when the stored weights are not in pb).

    python add_rate_block.py <ENERGY_TAG> [--sigma-prior PB] [--mom DIR] [--prior DIR]

Writes 'rate' into moments_<E>/lambda_export.json and, per scheme, into
moments_<E>/lambda_export_variations.json.  Idempotent.
"""
import os, sys, json, argparse, csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("energy")
ap.add_argument("--sigma-prior", type=float, default=None, help="prior total cross section in pb (overrides the mean stored weight)")
ap.add_argument("--mom", default=None); ap.add_argument("--prior", default=None)
ap.add_argument("--prior-weights-are-pb", action="store_true", help="the stored per-event weights are in pb (sigma = sum of weights / number GENERATED); without this flag or --sigma-prior, K is left null")
ap.add_argument("--prior-n-generated", type=int, default=None, help="number of events the prior sample was generated/showered from, when the stored file holds only those passing a cut. POWHEG unweighted events all carry the same |w| = sigma_gen, so the MEAN over a cut subset returns the generation cross section, not the cross section of the selection: sigma = sum(w)/N_generated is the right estimator. Defaults to the number of stored events.")
a = ap.parse_args()
E = a.energy
MOM = a.mom or f"moments_{E}"; PRIOR = a.prior or f"sherpa_prior_{E}"
CSV = f"{MOM}/DYMoments_N4LLp+N3LO.csv"

# --- calculation: total rate per scheme (the normalization moment) --------------------------
def scheme_name(fo, res):
    if fo.startswith("CV") and res.startswith("CV"): return "central"
    return fo.split("->")[0] if not fo.startswith("CV") else res.split("->")[0]
sig = {}; unc = {}
for r in csv.DictReader(open(CSV)):
    if r["O1"] == "dphi^0" and r["O2"] == "lndphi^0":
        nm = scheme_name(r["ScaleFO"], r["ScaleRes"]); sig[nm] = float(r["value"]); unc[nm] = float(r["uncertainty"])
assert "central" in sig, "normalization moment (dphi^0 x lndphi^0, CV/CV) not found"
Z, dZ = sig["central"], unc["central"]

# --- prior: mean stored event weight -------------------------------------------------------
import pandas as pd
w = pd.read_csv(f"{PRIOR}/pT_weight.csv.gz", header=None, low_memory=False).iloc[:, 0].to_numpy(dtype=float)
N = int(len(w)); mean_w = float(w.mean())
NGEN = a.prior_n_generated or N
if a.sigma_prior is not None:
    sp, sp_src = a.sigma_prior, "user-supplied total cross section of the prior sample"
elif a.prior_weights_are_pb:
    sp = float(w.sum()) / NGEN
    sp_src = (f"stored per-event weights in pb: sigma = sum(w)/N_generated = {w.sum():.6g}/{NGEN:,}"
              + ("" if NGEN == N else f"; the file holds {N:,} events passing the analysis cut, so dividing by N "
                                      "instead would return the generation cross section, not the cross section of the selection"))
else:
    sp, sp_src = None, ("not set: the stored per-event weights are not certified to be in pb; supply --sigma-prior "
                        "(the sample's total cross section in pb for m_ll >= 40 GeV) to obtain K")
K = (Z / sp) if sp else None
print(f"  [{E}] sigma_calc = {Z:.3f} +- {dZ:.3f} pb (MC);  mean stored weight = {mean_w:.4f} (N={N:,});  sigma_prior = {sp};  K = {K}")
per = {nm: {"sigma_calc_pb": sig[nm], "sigma_calc_mc_unc_pb": unc[nm], "K": (sig[nm] / sp) if sp else None} for nm in sig}
if K:
    ks = np.array([per[nm]["K"] for nm in per if nm != "central"])
    print(f"  [{E}] scheme K range: {ks.min():.5f} .. {ks.max():.5f}  ({100*(ks.min()/K-1):+.2f}% / {100*(ks.max()/K-1):+.2f}% about central)")

block = {
    "description": ("Overall rate normalization. The MaxEnt weight preserves the prior's normalization "
                    "(sum w = sum w0) and transfers only the shape; multiply every weight (central branch and "
                    "tail branch alike) by K to normalize the sample to the calculation's total rate. "
                    "Per scheme use per_scheme[s].K with schemes[s] of lambda_export_variations.json."),
    "phase_space": "inclusive, m_ll >= 40 GeV, all lepton rapidities (the calculation's normalization moment); the prior sample is generated with the same mass cut",
    "sigma_calc_pb": Z, "sigma_calc_mc_unc_pb": dZ,
    "sigma_prior_pb": sp, "sigma_prior_source": sp_src, "prior_n_events": N, "prior_n_generated": NGEN, "prior_mean_stored_weight": mean_w,
    "K": K,
    "per_scheme": per,
}
for fn in ("lambda_export.json", "lambda_export_variations.json"):
    p = f"{MOM}/{fn}"; d = json.load(open(p))
    d["rate"] = block
    if "schemes" in d:
        for nm, s in d["schemes"].items():
            if nm in per and per[nm]["K"] is not None: s["K"] = per[nm]["K"]
            else: s.pop("K", None)                      # no certified K: leave no stale value behind
    json.dump(d, open(p, "w"), indent=1)
    print(f"  wrote rate block -> {p}")
