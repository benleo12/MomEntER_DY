"""7-point prior-scale band, plotted the way the weights are actually meant to be used.

For each of the 7 mu_R/mu_F points (central + 6 variations) the event weight is the
gated blend

    w_V = beta * [ w0_V * exp(sum_k lambda_V[k] phi_k - C_V) ]  +  (1 - beta) * w0_V

with beta the smootherstep gate over qT in [GATE_LO, GATE_HI].  So

  * below the gate  (beta = 1)  the varied prior is reweighted onto the theory target,
    and the seven samples collapse -- what is left is the residual of the finite moment
    basis, not a scale uncertainty;
  * above the gate  (beta = 0)  the weights reduce to the bare prior variation, so the
    band there IS the generator's own mu_R/mu_F uncertainty.

That hand-off is the point: the calculation supplies the uncertainty where it is valid,
the generator supplies it where the calculation stops.

Usage:
    ENERGY=13TeV_v3 VARJSON=moments_13TeV_v3/lambda_export_prior_variations_MAIN.json \
    VARDIR=variations python make_7point_band.py [out.pdf]

Env:
    ENERGY   prior/moments tag (default 13TeV_v3)
    VARJSON  prior-variation lambda file (default: the MAIN set if present, else the All set)
    VARDIR   subdirectory of the prior holding the variation weight columns
    NEV      number of events (default: all)
"""
import os, sys, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"4")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import optimizer_DY_unc as o
from apply_lambdas import reweight, reweight_scheme

ENE   = os.environ.get("ENERGY", "13TeV_v3")
PRIOR = f"sherpa_prior_{ENE}"; MOM = f"moments_{ENE}"
CEN   = f"{MOM}/lambda_export.json"
_main = f"{MOM}/lambda_export_prior_variations_MAIN.json"
VARJ  = os.environ.get("VARJSON", _main if os.path.exists(_main) else f"{MOM}/lambda_export_prior_variations.json")
VARD  = os.environ.get("VARDIR", "variations")
OUT   = sys.argv[1] if len(sys.argv) > 1 else "fig_7point_band.pdf"

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 13, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "figure.dpi": 160, "lines.linewidth": 1.6,
})

p  = o.load_prior(PRIOR); N = min(len(p["w"]), int(os.environ.get("NEV", "10"*9)))
qT = p["pT"][:N].astype(float); dd = p["d"][:N].astype(float)
mm = p["m"][:N].astype(float); w0 = p["w"][:N].astype(float); del p
gate = json.load(open(CEN))["gating"]
GLO, GHI = gate["window_GeV"]

E = np.array([0,2,4,6,8,10,13,16,20,25,30,37,45,55,65,75,85,105,125,150,175,200,250,300,400,600,900.])
def hu(w):
    h,_ = np.histogram(qT, bins=E, weights=w); s = h/np.diff(E)
    return s/np.sum(s*np.diff(E))

CACHE = OUT.replace(".pdf", "_hists.npz")
schemes = list(json.load(open(VARJ))["schemes"])
if os.path.exists(CACHE) and not int(os.environ.get("NOCACHE", "0")):
    _z = np.load(CACHE, allow_pickle=True); cen = _z["cen"]; H = list(_z["H"]); schemes = list(_z["schemes"])
    print(f"  loaded cached histograms from {CACHE}")
else:
    cen = hu(reweight(w0, qT, mm, dd, jpath=CEN, gate=True))
    H = []
    for V in schemes:
        wv = pd.read_csv(f"{PRIOR}/{VARD}/{V}.csv.gz", header=None).values.ravel()[1:][:N].astype(float)
        w  = reweight_scheme(wv, qT, mm, dd, scheme=V, jpath=VARJ, gate=True)
        ne = 100*w.sum()**2/(N*(w*w).sum())
        print(f"  {V:22s} N_eff={ne:7.3f}%", flush=True)
        H.append(hu(w))
    np.savez(CACHE, cen=cen, H=np.array(H), schemes=np.array(schemes))
    print(f"  cached histograms -> {CACHE}")
M   = np.vstack(H + [cen])
lo, hi = M.min(0), M.max(0)
band   = 100*0.5*(hi-lo)/cen
ctr    = 0.5*(E[:-1]+E[1:])

fig, (ax, ar) = plt.subplots(2, 1, figsize=(6.8, 6.4), sharex=True,
                             gridspec_kw={"height_ratios":[1.7,1.0], "hspace":0.05})
ee = E
_lab = {"MUR_0.5__MUF_0.5":r"$(\tfrac12,\tfrac12)$","MUR_0.5__MUF_1":r"$(\tfrac12,1)$",
        "MUR_1__MUF_0.5":r"$(1,\tfrac12)$","MUR_1__MUF_2":r"$(1,2)$",
        "MUR_2__MUF_1":r"$(2,1)$","MUR_2__MUF_2":r"$(2,2)$"}
_col = ["#e42536","#f89c20","#964a8b","#3f90da","#92dadd","#a96b59"]
ax.stairs(hi, ee, baseline=lo, fill=True, color="0.6", alpha=0.30, lw=0,
          label=r"7-point $\mu_R,\mu_F$ envelope")
for j, V in enumerate(schemes):
    ax.stairs(H[j], ee, color=_col[j % len(_col)], lw=1.0, alpha=0.9,
              label=_lab.get(str(V), str(V)))
ax.stairs(cen, ee, color="k", lw=1.8, label=r"central $(1,1)$")
ax.axvspan(GLO, GHI, color="0.5", alpha=0.10, lw=0)
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_ylabel(r"$1/\sigma\;\mathrm{d}\sigma/\mathrm{d}q_T$")
ax.legend(loc="lower left", fontsize=8.5, ncol=2, handlelength=1.4)
ax.set_title(rf"7-point prior-scale band, gated hand-off over ${GLO:.0f}$–${GHI:.0f}$ GeV",
             fontsize=11, pad=6)

ar.axhline(1, color="k", lw=0.7)
ar.stairs(hi/cen, ee, baseline=lo/cen, fill=True, color="0.6", alpha=0.30, lw=0)
for j, V in enumerate(schemes):
    ar.stairs(H[j]/cen, ee, color=_col[j % len(_col)], lw=1.1, alpha=0.9)
ar.axvspan(GLO, GHI, color="0.5", alpha=0.10, lw=0)
ar.set_xscale("log"); ar.set_xlabel(r"$q_T$ [GeV]")
ar.set_ylabel("ratio to central"); ar.set_ylim(0.5, 1.6)
ar.text(0.5*(GLO+GHI), 0.55, "hand-off", color="0.45", fontsize=8, ha="center")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"  wrote {OUT} (+ .png)   [{os.path.basename(VARJ)}, {VARD}]")
m = ctr < GLO
print(f"  bulk (qT<{GLO:.0f}) median {np.median(band[m]):.3f}%   "
      f"tail (qT>{GHI:.0f}) median {np.median(band[ctr>GHI]):.3f}%   max {band.max():.3f}%")
