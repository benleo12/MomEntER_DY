"""7-point prior-scale variations for all three constrained observables, gated blend.

Per mu_R/mu_F point V the event weight is

    w_V = beta * [ w0_V * exp(sum_k lambda_V[k] phi_k - C_V) ] + (1 - beta) * w0_V

beta = smootherstep gate over qT in [GATE_LO, GATE_HI]. Below the gate the varied priors
are reweighted onto the theory target and collapse; above it the weights reduce to the bare
prior variation, so the spread there is the generator's own muR/muF uncertainty.

Three observables: qT, rT = qT/m_ll, and the acoplanarity d = pi - dphi_ll.

Usage:  ENERGY=13TeV_v3 VARDIR=variations python make_7point_band3.py [out.pdf]
"""
import os, sys, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"4")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import optimizer_DY_unc as o
from apply_lambdas import reweight, reweight_scheme

ENE   = os.environ.get("ENERGY", "13TeV_v3")
PRIOR = f"sherpa_prior_{ENE}"; MOM = f"moments_{ENE}"
CEN   = f"{MOM}/lambda_export.json"
_main = f"{MOM}/lambda_export_prior_variations_MAIN.json"
VARJ  = os.environ.get("VARJSON", _main if os.path.exists(_main) else f"{MOM}/lambda_export_prior_variations.json")
VARD  = os.environ.get("VARDIR", "variations")
OUT   = sys.argv[1] if len(sys.argv) > 1 else "fig_7point_band3.pdf"

# Rivet-like house style (same as the End Matter figure)
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 12, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3, "ytick.minor.size": 3,
    "legend.frameon": False, "figure.dpi": 160, "lines.linewidth": 1.4,
})

p  = o.load_prior(PRIOR); N = min(len(p["w"]), int(os.environ.get("NEV", "10"*9)))
qT = p["pT"][:N].astype(float); dd = p["d"][:N].astype(float)
mm = p["m"][:N].astype(float); rt = p["rT"][:N].astype(float); w0 = p["w"][:N].astype(float); del p
GLO, GHI = json.load(open(CEN))["gating"]["window_GeV"]

OBS = [("qT", qT, np.array([0,2,4,6,8,10,13,16,20,25,30,37,45,55,65,75,85,105,125,150,175,200,250,300,400,600,900.]),
        r"$q_T$ [GeV]", True),
       ("rT", rt, np.concatenate([np.linspace(0,0.5,21), np.array([0.6,0.75,0.9,1.1,1.4,1.8,2.4,3.2,4.5,6.5,9.0])]),
        r"$r_T = q_T/m_{\ell\ell}$", False),
       ("aco", dd, np.concatenate([np.linspace(0,1.0,21), np.array([1.1,1.25,1.4,1.6,1.8,2.0,2.25,2.5,2.8,3.15])]),
        r"acoplanarity $d = \pi - \Delta\phi_{\ell\ell}$", False)]

def hu(x, w, E):
    h,_ = np.histogram(x, bins=E, weights=w); s = h/np.diff(E)
    return s/np.sum(s*np.diff(E))

CACHE = OUT.replace(".pdf", "_hists.npz")
schemes = list(json.load(open(VARJ))["schemes"])
if os.path.exists(CACHE) and not int(os.environ.get("NOCACHE", "0")):
    _z = np.load(CACHE, allow_pickle=True)
    CEN_H = {k: _z[f"cen_{k}"] for k,_,_,_,_ in OBS}
    VAR_H = {k: _z[f"var_{k}"] for k,_,_,_,_ in OBS}
    schemes = list(_z["schemes"])
    print(f"  loaded cached histograms from {CACHE}")
else:
    wc = reweight(w0, qT, mm, dd, jpath=CEN, gate=True)
    WV = []
    for V in schemes:
        wv = pd.read_csv(f"{PRIOR}/{VARD}/{V}.csv.gz", header=None).values.ravel()[1:][:N].astype(float)
        w  = reweight_scheme(wv, qT, mm, dd, scheme=V, jpath=VARJ, gate=True)
        print(f"  {V:22s} N_eff={100*w.sum()**2/(N*(w*w).sum()):7.3f}%", flush=True)
        WV.append(w)
    CEN_H = {k: hu(x, wc, E) for k,x,E,_,_ in OBS}
    VAR_H = {k: np.vstack([hu(x, w, E) for w in WV]) for k,x,E,_,_ in OBS}
    np.savez(CACHE, schemes=np.array(schemes),
             **{f"cen_{k}": CEN_H[k] for k,_,_,_,_ in OBS},
             **{f"var_{k}": VAR_H[k] for k,_,_,_,_ in OBS})
    print(f"  cached histograms -> {CACHE}")

LAB = {"MUR_0.5__MUF_0.5":r"$(\frac{1}{2},\frac{1}{2})$","MUR_0.5__MUF_1":r"$(\frac{1}{2},1)$",
       "MUR_1__MUF_0.5":r"$(1,\frac{1}{2})$","MUR_1__MUF_2":r"$(1,2)$",
       "MUR_2__MUF_1":r"$(2,1)$","MUR_2__MUF_2":r"$(2,2)$"}
COL = ["#e42536","#f89c20","#964a8b","#3f90da","#92dadd","#a96b59"]

fig = plt.figure(figsize=(6.6, 11.4))
outer = fig.add_gridspec(3, 1, hspace=0.34)
for i,(key, x, E, xlabel, logx) in enumerate(OBS):
    g  = outer[i].subgridspec(2, 1, height_ratios=[1.7,1.0], hspace=0.05)
    ax = fig.add_subplot(g[0]); ar = fig.add_subplot(g[1], sharex=ax)
    cen = CEN_H[key]; Hs = VAR_H[key]
    M = np.vstack([Hs, cen]); lo, hi = M.min(0), M.max(0)
    if key == "qT":
        ax.axvspan(GLO, GHI, color="0.5", alpha=0.10, lw=0); ar.axvspan(GLO, GHI, color="0.5", alpha=0.10, lw=0)
    ax.stairs(hi, E, baseline=lo, fill=True, color="0.6", alpha=0.30, lw=0,
              label=r"7-point envelope" if i==0 else None)
    for j,V in enumerate(schemes):
        ax.stairs(Hs[j], E, color=COL[j%len(COL)], lw=1.0, alpha=0.9,
                  label=LAB.get(str(V), str(V)) if i==0 else None)
    ax.stairs(cen, E, color="k", lw=1.8, label=r"central $(1,1)$" if i==0 else None)
    ax.set_yscale("log"); ax.set_ylabel(r"$1/\sigma\;\mathrm{d}\sigma/\mathrm{d}x$")
    if logx: ax.set_xscale("log"); ax.set_xlim(max(E[0],1.0), E[-1])
    else:    ax.set_xlim(E[0], E[-1])
    plt.setp(ax.get_xticklabels(), visible=False)
    if i == 0:
        ax.legend(loc="lower left", fontsize=8, ncol=2, handlelength=1.4)
        ax.set_title(rf"7-point prior scales, gated hand-off {GLO:.0f}–{GHI:.0f} GeV "
                     rf"({ENE}, {N/1e6:.1f}M)", fontsize=10.5, pad=6)
    ar.axhline(1, color="k", lw=0.7)
    ar.stairs(hi/cen, E, baseline=lo/cen, fill=True, color="0.6", alpha=0.30, lw=0)
    for j,V in enumerate(schemes):
        ar.stairs(Hs[j]/cen, E, color=COL[j%len(COL)], lw=1.1, alpha=0.9)
    ar.set_ylabel("ratio to central"); ar.set_ylim(0.6, 1.5)
    ar.yaxis.set_major_locator(MultipleLocator(0.2)); ar.set_xlabel(xlabel)
    if logx: ar.set_xscale("log"); ar.set_xlim(max(E[0],1.0), E[-1])
    else:    ar.set_xlim(E[0], E[-1]); ar.xaxis.set_minor_locator(AutoMinorLocator())
    ctr = 0.5*(E[:-1]+E[1:]); band = 100*0.5*(hi-lo)/cen
    if key == "qT":
        print(f"  {key}: bulk median {np.median(band[ctr<GLO]):.2f}%  tail median {np.median(band[ctr>GHI]):.2f}%")
    else:
        print(f"  {key}: median {np.median(band):.2f}%  max {band.max():.2f}%")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"  wrote {OUT} (+ .png)   [{os.path.basename(VARJ)}, {VARD}]")
