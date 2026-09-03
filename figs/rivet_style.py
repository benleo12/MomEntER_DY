"""Rivet make-plots house style (Stefan's figures): log-x over the full range, log-y spectrum on top with the
reference (data or calculation) as black points, MC as step histograms with shaded uncertainty bands, and a
MC/Data (MC/Theory) panel spanning 0.6-1.4 underneath.  Everything is drawn from the npz files written on
Perlmutter, so the style can be iterated without touching the samples.

  python rivet_style.py fid <FIG_ATLASFID_pwg186p_hists.npz> <outprefix>
  python rivet_style.py thy <PAPER_HISTS_<tag>.npz> <outprefix> "<prior label>" "<title>"
Writes one figure per observable (<outprefix>_<obs>) and a vertical stack (<outprefix>_stack, obs listed in STACK).
"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator

# Rivet make-plots style (Stefan's default.mplstyle: Palatino/usetex, hairline axes,
# inward ticks, Rivet color cycle, frameless legend).  Looked up next to this script;
# override with MPLSTYLE=/path/to/default.mplstyle.  USETEX=0 for LaTeX-less hosts.
_STYLE = os.environ.get("MPLSTYLE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.mplstyle"))
plt.style.use(_STYLE)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}\\usepackage{amssymb}"
if os.environ.get("USETEX", "1") != "1": plt.rcParams["text.usetex"] = False
# ScottPlot 4.1 default palette = Category10 (matplotlib tab10): the group's usual,
# colour-blind-friendlier choice.  Prior = C10 red (dashed), reweighted = C10 blue.
from cycler import cycler as _cyc
C10 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
plt.rcParams["axes.prop_cycle"] = _cyc(color=C10)
RED, BLUE, GREY = "#d62728", "#1f77b4", "0.55"
PRI, REW = "#AEC7E8", "#D62728"     # Stefan's convention: prior light blue solid, prediction red solid
RLO, RHI = 0.5, 1.4999                                           # ratio panel as in Stefan's plots
STACK = os.environ.get("STACK", "q,d").split(",")
THY_LABEL = r"N$^4$LL$^\prime$+N$^3$LO"

def _knob(lbl): return lbl[3:] if lbl.startswith("0p5") else (lbl[1:] if lbl.startswith("2") else lbl)

def dens(h, e, tot=None):
    """density normalised to the FULL-range total tot (default: this histogram's own total)"""
    return h / np.diff(e) / (np.nansum(h) if tot is None else tot)

def scheme_band(H, e, schs, tot=None):
    """H: (nbins, 2+nsch) prior|central|schemes -> central density and per-knob quadrature band."""
    tot = np.nansum(H, axis=0) if tot is None else tot
    rew = dens(H[:, 1], e, tot[1]); kn = {}
    for j, s in enumerate(schs): kn.setdefault(_knob(s), []).append(dens(H[:, 2 + j], e, tot[2 + j]))
    up2, dn2 = np.zeros_like(rew), np.zeros_like(rew)
    for vs in kn.values():
        up2 += np.maximum.reduce([np.maximum(v - rew, 0) for v in vs])**2
        dn2 += np.maximum.reduce([np.maximum(rew - v, 0) for v in vs])**2
    return rew, rew - np.sqrt(dn2), rew + np.sqrt(up2)

def rebin(e, y, e2, density=True):
    idx = np.searchsorted(e, e2); assert np.allclose(e[idx], e2), "e2 not a subset of e"
    w = np.diff(e); out = np.zeros(len(e2) - 1)
    for j in range(len(out)):
        sl = slice(idx[j], idx[j + 1]); out[j] = np.sum(y[sl] * w[sl]) / (e2[j + 1] - e2[j]) if density else np.sum(y[sl])
    return out

def snap(e_fine, targets):
    out = [e_fine[0]]
    for t in targets:
        k = e_fine[np.argmin(np.abs(e_fine - t))]
        if k > out[-1] + 1e-12 and k < e_fine[-1] - 1e-12: out.append(k)
    out.append(e_fine[-1]); return np.array(out)

# ---------------- frames -------------------------------------------------------------------------------
def frames(n, figsize_one=(4.67, 4.68), gap=0.28):
    """n stacked (spectrum, ratio) pairs; returns fig, [(ax, ar), ...]"""
    fig = plt.figure(figsize=(figsize_one[0], figsize_one[1] * n))
    outer = fig.add_gridspec(n, 1, hspace=gap, left=0.1875, right=0.968, top=1 - 0.07 / n, bottom=0.11 / n)
    pairs = []
    for i in range(n):
        gs = outer[i].subgridspec(2, 1, height_ratios=[6.0, 4.0], hspace=0.0)
        ax = fig.add_subplot(gs[0]); ar = fig.add_subplot(gs[1], sharex=ax)
        ax.set_yscale("log"); plt.setp(ax.get_xticklabels(), visible=False)
        ar.set_ylim(RLO, RHI); ar.yaxis.set_major_locator(MultipleLocator(0.2)); ar.axhline(1.0, color="k", lw=0.6, zorder=1)
        pairs.append((ax, ar))
    return fig, pairs

def decorate(ax, ar, title, ylab, rlab, xlab, notes=(), corner="lower left"):
    ax.set_ylabel(ylab, loc="top"); ar.set_ylabel(rlab); ar.set_xlabel(xlab, loc="right")
    ax.set_title(title, loc="left")
    for k, txt in enumerate(notes):
        if corner == "lower left": ax.text(0.04, 0.32 + 0.09 * (len(notes) - 1 - k), txt, transform=ax.transAxes, ha="left", va="bottom")
        else:                      ax.text(0.96, 0.62 - 0.09 * k, txt, transform=ax.transAxes, ha="right", va="top")

def steps(ax, e, y, **kw): ax.stairs(y, e, baseline=None, **kw)
def bandfill(ax, e, lo, hi, color, alpha=0.30, hatch=None, **kw):
    if hatch: ax.stairs(hi, e, baseline=lo, fill=False, edgecolor=color, lw=0, hatch=hatch, **kw)
    else:     ax.stairs(hi, e, baseline=lo, fill=True, color=color, alpha=alpha, lw=0, **kw)
def points(ax, e, y, err, **kw):
    c = 0.5 * (e[:-1] + e[1:]); ax.errorbar(c, y, yerr=err, xerr=[c - e[:-1], e[1:] - c], fmt="o", ms=2, lw=1, color="k", capsize=0, zorder=25, **kw)
def legend_first(ax, first, **kw):
    h, l = ax.get_legend_handles_labels(); o = [l.index(first)] + [i for i in range(len(l)) if l[i] != first]
    ax.legend([h[i] for i in o], [l[i] for i in o], **kw)
def finish(fig, out):
    for ext in (".pdf", ".png"): fig.savefig(out + ext, dpi=220)
    plt.close(fig); print(f"  wrote {out}.pdf/.png")

# ---------------- one observable, reference = data or theory ---------------------------------------------
def draw(ax, ar, e, ref, ref_err, pri, rew, rlo, rhi, prior_label, ref_label, ref_band=None, ref_hatch=None, xscale="log", xlo=None, rew_label="MaxEnt reweighted", corner="lower left", xhi=None):
    """ref/ref_err: reference (points); ref_band: (lo,hi) shaded reference band (theory scale); ref_hatch: (lo,hi) hatched
    (theory MC-stat).  MC bins with no sample content are masked."""
    good = np.isfinite(ref) & (ref > 0); has = good & (pri > 0.05 * np.where(good, ref, 1))   # MC bins with <5% of the reference content are not a comparison
    nan = lambda y, m: np.where(m, y, np.nan)
    if ref_band is not None: bandfill(ax, e, nan(ref_band[0], good), nan(ref_band[1], good), GREY, alpha=0.35)
    points(ax, e, nan(ref, good), nan(ref_err, good), label=ref_label)
    steps(ax, e, nan(pri, has), color=PRI, lw=1, zorder=6, label=prior_label)
    bandfill(ax, e, nan(rlo, has), nan(rhi, has), REW, alpha=0.20, zorder=7); steps(ax, e, nan(rew, has), color=REW, lw=1, zorder=7, label=rew_label)
    r = lambda y, m: np.where(m, y / np.where(good, ref, 1), np.nan)
    if ref_band is not None: bandfill(ar, e, r(ref_band[0], good), r(ref_band[1], good), GREY, alpha=0.35, label="scale")
    else:                    bandfill(ar, e, r(ref - ref_err, good), r(ref + ref_err, good), "0.75", alpha=0.5)
    if ref_hatch is not None: bandfill(ar, e, r(ref_hatch[0], good), r(ref_hatch[1], good), "0.35", hatch="////", label="MC stat.")
    points(ar, e, r(ref, good), r(ref_err, good))
    steps(ar, e, r(pri, has), color=PRI, lw=1, zorder=6); bandfill(ar, e, r(rlo, has), r(rhi, has), REW, alpha=0.20, zorder=7); steps(ar, e, r(rew, has), color=REW, lw=1, zorder=7)
    if xscale == "log": ax.set_xscale("log"); ax.set_xlim(xlo if xlo is not None else e[0], xhi if xhi is not None else e[-1])
    else: ax.set_xlim(e[0], xhi if xhi is not None else e[-1])
    ymin = np.nanmin(np.concatenate([ref[good], rew[has], pri[has]])); L = np.log10(ymin * 0.3)
    if np.ceil(L) - L < 0.35: L = np.ceil(L) - 0.6                         # keep the lowest decade label clear of the ratio panel's 1.4
    ax.set_ylim(10**L, np.nanmax(ref[good]) * 4)
    if corner == "lower left": legend_first(ax, ref_label, alignment="left", loc="lower left", bbox_to_anchor=(0.01, 0.01), markerfirst=True)
    else:                      legend_first(ax, ref_label, alignment="right", loc="upper right", bbox_to_anchor=(1.0, 0.97), markerfirst=False)
    if ref_band is not None: ar.legend(loc="lower right", ncol=2, fontsize=8, handlelength=1.6, markerfirst=False)
    return dict(prior_med=100*np.median(np.abs(pri[has]/ref[has]-1)), prior_max=100*np.max(np.abs(pri[has]/ref[has]-1)),
                rew_med=100*np.median(np.abs(rew[has]/ref[has]-1)), rew_max=100*np.max(np.abs(rew[has]/ref[has]-1)), nbins=int(has.sum()))

# ---------------- POWHEG fiducial vs ATLAS -------------------------------------------------------------------
FID = {"qt": dict(xl=r"$p_\mathrm{T}^{\ell\ell}$ [GeV]", yl=r"$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}p_\mathrm{T}^{\ell\ell}$", xlo=0.5),
       "ps": dict(xl=r"$\phi^*_\eta$", yl=r"$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}\phi^*_\eta$", xlo=2e-3)}
FID_TITLE = {"qt": "Transverse momentum of the lepton pair", "ps": r"$\phi^*_\eta$ of the lepton pair"}
FID_NOTES = (r"$pp\to l^+l^-$, dressed", r"$p_{T,l}\ge 27$ GeV, $|\eta_l|\le 2.5$", r"$66$ GeV$\le m_{ll}\le 116$ GeV")
DATA_LABEL = "ATLAS Data, EPJC80(2020)616"
def rew_label_for(prior_label): return r"N$^4$LL$^\prime$+N$^3$LO+POWHEG" if "POWHEG" in prior_label else r"N$^4$LL$^\prime$+N$^3$LO+MEPS@NLO"

def fid_one(ax, ar, z, key, prior_label):
    e = z[f"{key}_edges"]; decorate(ax, ar, FID_TITLE[key], FID[key]["yl"], "Theory~/~Data", FID[key]["xl"], notes=FID_NOTES)
    return draw(ax, ar, e, z[f"{key}_dat"], z[f"{key}_err"], z[f"{key}_pri"], z[f"{key}_rew"], z[f"{key}_rlo"], z[f"{key}_rhi"],
                prior_label, DATA_LABEL, xlo=FID[key]["xlo"], rew_label=rew_label_for(prior_label), xhi=(900.0 if key == "qt" else None))

def fid(npz, pre, prior_label="POWHEG Prior"):
    z = np.load(npz); res = {}
    for key in ("qt", "ps"):
        fig, [(ax, ar)] = frames(1); res[key] = fid_one(ax, ar, z, key, prior_label); finish(fig, f"{pre}_{key}")
        e, dat, pri, rew = z[f"{key}_edges"], z[f"{key}_dat"], z[f"{key}_pri"], z[f"{key}_rew"]
        m = (dat > 0) & (pri > 0) & (0.5 * (e[:-1] + e[1:]) < 200)
        print(f"  {key}: <200 GeV: prior/data med {100*np.median(np.abs(pri[m]/dat[m]-1)):.2f}%  rew/data med {100*np.median(np.abs(rew[m]/dat[m]-1)):.2f}%; full range: rew med {res[key]['rew_med']:.2f}% max {res[key]['rew_max']:.1f}%")
    fig, pairs = frames(2)
    for (ax, ar), key in zip(pairs, ("qt", "ps")): fid_one(ax, ar, z, key, prior_label)
    finish(fig, f"{pre}_stack"); json.dump(res, open(f"{pre}_summary.json", "w"), indent=1)

# ---------------- inclusive prior vs N4LL'+N3LO --------------------------------------------------------------
THY_TITLE = {"q": "Transverse momentum of the lepton pair", "r": "Scaled transverse momentum of the lepton pair", "d": "Acoplanarity of the lepton pair"}
THY = {"q": dict(xl=r"$q_\mathrm{T}=p_\mathrm{T}^{\ell\ell}$ [GeV]", yl=r"$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}q_\mathrm{T}$ [1/GeV]", xlo=0.5, xs="log"),
       "r": dict(xl=r"$r_\mathrm{T}=q_\mathrm{T}/m_{\ell\ell}$", yl=r"$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}r_\mathrm{T}$", xlo=0.005, xs="log"),
       "d": dict(xl=r"acoplanarity $\;d=\pi-\Delta\phi_{\ell\ell}$", yl=r"$1/\sigma\,\mathrm{d}\sigma/\mathrm{d}d$", xlo=None, xs="lin")}
ATLAS_QT = np.array([2,4,6,8,10,12,14,16,18,20,22.5,25,27.5,30,33,36,39,42,45,48,51,54,57,61,65,70,75,80,85,95,105,125,150,175,200,250,300,350,400,470,550,650,900])

def thy_hists(z, key):
    e, H = z[f"{key}_e"], z[f"{key}_H"]; T = {k: z[f"{key}_{k}"] for k in ("cen", "slo", "shi", "stlo", "sthi")}
    if key == "q":                                       # theory (normalised over 0-200) ends at 200 GeV: MC normalised over the same range
        keep = e <= 200 + 1e-9; e2 = e[keep]; H = H[: len(e2) - 1]; T = {k: v[: len(e2) - 1] for k, v in T.items()}; e = e2; tot = np.nansum(H, axis=0)
    else:                                                # theory normalised over its full range (r_T 0-5, d 0-2.51): MC likewise, then r_T truncated at 200 GeV/m_Z for display
        tot = np.nansum(H, axis=0)
        e2 = snap(e[e <= 200 / 91.19 + 0.005], ATLAS_QT[ATLAS_QT <= 200] / 91.19) if key == "r" else (e[::4] if np.isclose(e[-1], e[::4][-1]) else np.append(e[::4], e[-1]))
        H = np.column_stack([rebin(e, H[:, c], e2, density=False) for c in range(H.shape[1])]); T = {k: rebin(e, v, e2) for k, v in T.items()}; e = e2
    return e, H, T, tot

def thy_one(ax, ar, z, key, prior_label, title):
    schs = [str(s) for s in z["schemes"]]; e, H, T, tot = thy_hists(z, key)
    pri = dens(H[:, 0], e, tot[0]); rew, rlo, rhi = scheme_band(H, e, schs, tot)
    corner = "upper right" if key == "d" else "lower left"
    decorate(ax, ar, THY_TITLE[key], THY[key]["yl"], "Ratio to calculation", THY[key]["xl"], notes=(title, r"$m_{ll}\ge 40$ GeV"), corner=corner)
    return draw(ax, ar, e, T["cen"], 0.5 * (T["sthi"] - T["stlo"]), pri, rew, rlo, rhi, prior_label, THY_LABEL,
                ref_band=(T["slo"], T["shi"]), ref_hatch=(T["stlo"], T["sthi"]), xscale=THY[key]["xs"], xlo=THY[key]["xlo"], rew_label=rew_label_for(prior_label), corner=corner)

def thy(npz, pre, prior_label, title):
    z = np.load(npz, allow_pickle=True); res = {}
    for key in ("q", "r", "d"):
        fig, [(ax, ar)] = frames(1); res[key] = thy_one(ax, ar, z, key, prior_label, title); finish(fig, f"{pre}_{key}")
        print(f"  {key}: prior/theory med {res[key]['prior_med']:.2f}% max {res[key]['prior_max']:.1f}%   rew/theory med {res[key]['rew_med']:.2f}% max {res[key]['rew_max']:.1f}%  ({res[key]['nbins']} bins)")
    fig, pairs = frames(len(STACK))
    for (ax, ar), key in zip(pairs, STACK): thy_one(ax, ar, z, key, prior_label, title)
    finish(fig, f"{pre}_stack")
    res["neff"] = 100 * float(z["neff"]); res["N"] = int(z["N"]); res["gated"] = bool(z["gated"]); res["K"] = len(z["names"])
    print(f"  N_eff {res['neff']:.2f}%  N={res['N']:,}  gated={res['gated']}  K={res['K']}")
    json.dump(res, open(f"{pre}_summary.json", "w"), indent=1)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fid": fid(sys.argv[2], sys.argv[3], *sys.argv[4:5])
    elif mode == "thy": thy(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
