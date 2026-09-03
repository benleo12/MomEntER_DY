"""Fig. 3 of the PRL (fig:calc): the N4LL'+N3LO calculation in the ATLAS fiducial phase
space of arXiv:1912.02844, compared with the measured normalized qT spectrum,
with the theory uncertainty decomposed into fixed-order, resummation, and
non-perturbative (Collins-Soper kernel, lattice-constrained) components.

  python make_fig1.py [outfile.pdf]
"""
import os, sys, re
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"2")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

OUT = sys.argv[1] if len(sys.argv) > 1 else "FIG1_calc_vs_ATLAS.pdf"
MODE = sys.argv[2] if len(sys.argv) > 2 else "lin"   # lin | log | split
ACC = "N4LL'+N3LO"
# Fiducial qT with per-bin MC-integration uncertainty (Wan-Li, mll in [66,116],
# pT_l>=27, |eta_l|<=2.5): 4 columns {lo, hi, value, uncertainty}.
THY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qT_fiducial_wunc_13TeV.m")
DAT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ATLAS_2019_I1768911.yoda")
# d04 = zpt combined BORN normalised: the calculation uses Born-level leptons (no QED
# FSR), so the Born table is the consistent comparison and matches Wan-Li's own plots.
# The dressed table (d27) differs by +2.2% in [0,2] GeV, decreasing with qT.
HIST = os.environ.get("ATLAS_HIST", "d04-x01-y01")
# NOTE: d27 = dressed leptons.  HEPData Table 4a is Born level and differs by
# ~2%.  Which is correct depends on the level of the fiducial calculation.

import os as _os
plt.style.use(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"default.mplstyle"))
plt.rcParams["figure.dpi"]=150; plt.rcParams["text.latex.preamble"]="\\usepackage{amsmath}\\usepackage{amssymb}"
from cycler import cycler as _cyc
C10=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
plt.rcParams["axes.prop_cycle"]=_cyc(color=C10)
BLUE,ORANGE,GREEN,PURPLE=C10[0],C10[1],C10[2],C10[4]

# ---------------------------------------------------------------- ATLAS data
def load_atlas(path, hist):
    """Read a Scatter2D from a Rivet reference .yoda: x, xerr-, xerr+, y, yerr-, yerr+."""
    rows, inblk = [], False
    for line in open(path, errors="ignore"):
        if f"BEGIN YODA_SCATTER2D_V2 /REF/ATLAS_2019_I1768911/{hist}" in line:
            inblk = True; continue
        if inblk and line.startswith("END YODA"): break
        if inblk:
            p = line.split()
            if len(p) >= 6 and (p[0][0].isdigit() or p[0][0] == "-"):
                try: rows.append([float(x) for x in p[:6]])
                except ValueError: pass
    a = np.array(rows); x, exl, exh, y, eyl, eyh = a.T
    return x - exl, x + exh, y, eyh, eyl

lo, hi, dat, dep, dem = load_atlas(DAT, HIST)
ctr, wid = 0.5 * (lo + hi), (hi - lo)
print(f"  ATLAS[{HIST}]: {len(dat)} bins, [{lo[0]:.0f},{hi[-1]:.0f}] GeV, "
      f"sum(f*dx)={np.sum(dat*wid):.4f}, median unc={100*np.median(dep/dat):.2f}%")

# ---------------------------------------------------------------- theory
def load_wju(path, acc):
    """Parse Wan-Li qT .m file.  Handles Mathematica '*^' exponents, which the
    shared loader does not (they appear in the fiducial file)."""
    txt = open(path, errors="ignore").read().replace("*^", "e")
    pat = (r'qTMat\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]\s*'
           r'=\s*\{((?:\s*\{[^}]+\},?)+)\s*\}')
    out = {}
    for m in re.finditer(pat, txt):
        a, res, fo = m.group(1), m.group(2), m.group(3)
        rows = re.findall(r"\{([^{}]+)\}", m.group(4))
        arr = []
        for r in rows:
            f = [x.strip() for x in r.split(",")]
            if len(f) < 3: continue
            try: arr.append([float(f[0]), float(f[1]), float(f[2]),
                             float(f[3]) if len(f) >= 4 else 0.0])
            except ValueError: pass
        if arr: out[(a, res, fo)] = np.array(arr)
    def norm(x): return x.replace("'", "p").replace("+", "")
    keys = [k for k in out if norm(k[0]) == norm(acc)]
    assert keys, f"accuracy {acc} not found; have {sorted(set(k[0] for k in out))}"
    cen = [k for k in keys if k[1].startswith("CV") and k[2].startswith("CV")]
    assert cen, "central (CV,CV) not found"
    c = out[cen[0]]
    edges = np.concatenate([c[:, 0], [c[-1, 1]]])
    vmap = {(k[1], k[2]): out[k][:, 2] for k in keys if k != cen[0]}
    return edges, c[:, 2], c[:, 3], vmap        # edges, central, central MC error, variations

te, tc, tc_unc, vmap = load_wju(THY, ACC)
print(f"  theory: {len(tc)} bins, {len(vmap)} variations, "
      f"qT range [{te[0]:.0f},{te[-1]:.0f}]  (real fiducial MC error)")

def rebin(src_edges, src_val, dst_edges):
    """integrate a density given on src_edges onto dst_edges (density out)."""
    out = np.zeros(len(dst_edges) - 1)
    for j in range(len(dst_edges) - 1):
        l = np.maximum(dst_edges[j],   src_edges[:-1])
        h = np.minimum(dst_edges[j+1], src_edges[1:])
        ov = np.clip(h - l, 0, None)
        out[j] = np.sum(src_val * ov) / (dst_edges[j+1] - dst_edges[j])
    return out

edges = np.concatenate([lo, [hi[-1]]])
def norm_to_unit(d):                       # normalize over the measured range
    return d / np.sum(d * wid)

th = norm_to_unit(rebin(te, tc, edges))

def rebin_err(src_edges, src_err, dst_edges):
    """Rebin an uncertainty, treating the source bins as independent."""
    out = np.zeros(len(dst_edges) - 1)
    for j in range(len(dst_edges) - 1):
        l = np.maximum(dst_edges[j],   src_edges[:-1])
        h = np.minimum(dst_edges[j+1], src_edges[1:])
        ov = np.clip(h - l, 0, None)
        out[j] = np.sqrt(np.sum((src_err * ov)**2)) / (dst_edges[j+1] - dst_edges[j])
    return out

# real per-bin MC-integration uncertainty, rebinned to the ATLAS binning
nv, ne = rebin(te, tc, edges), rebin_err(te, tc_unc, edges)
rel_num = np.where(nv > 0, ne / np.maximum(nv, 1e-300), 0.0)
k = (ctr < 200) & (nv > 0)
print(f"  numerical unc (real, fiducial): median "
      f"{100*np.median(rel_num[k]):.2f}%  max {100*rel_num[k].max():.2f}%")

# Band construction following Wan-Li: the 28 variations are paired into 14 knobs
# (x1/2 and x2 of each scale), and the per-knob deviations are added in QUADRATURE,
#   err_up^2 = sum_knob max(r_up-c, r_dn-c, 0)^2   (down analog),
# NOT as a min/max envelope. Each variation is normalized by its own sigma_tot.
def knob_of(k):
    """key = (res_label, fo_label); the knob is the non-CV label stripped of 0p5/2."""
    res, fo = k
    lab = fo if res.startswith("CV") and not fo.startswith("CV") else res
    return lab.replace("0p5", "").replace("2Mu", "Mu").replace("2Nu", "Nu") \
              .replace("2C0", "C0").replace("2kappa", "kappa")
knobs = {}
for k, v in vmap.items():
    knobs.setdefault(knob_of(k), []).append(norm_to_unit(rebin(te, np.array(v), edges)))
def grp_of(name):
    s = name.upper()
    if "C0_NP" in s: return "cs"
    if "KAPPA_NP" in s: return "kappa"
    if s.split("->")[0] in ("MUR", "MUF", "MURF"): return "fo"
    return "resum"
def quad(group=None):
    up2, dn2 = np.zeros_like(th), np.zeros_like(th)
    for name, vs in knobs.items():
        if group is not None and grp_of(name) != group: continue
        dev_up = np.maximum.reduce([np.maximum(v - th, 0) for v in vs])
        dev_dn = np.maximum.reduce([np.maximum(th - v, 0) for v in vs])
        up2 += dev_up**2; dn2 += dev_dn**2
    return th - np.sqrt(dn2), th + np.sqrt(up2)
print(f"    {len(knobs)} knobs paired from {len(vmap)} variations "
      f"({', '.join(sorted(set(grp_of(n) for n in knobs)))})")
lo_fo, hi_fo   = quad("fo")
lo_re, hi_re   = quad("resum")
lo_cs, hi_cs   = quad("cs")
lo_ka, hi_ka   = quad("kappa")
up2 = (hi_cs-th)**2 + (hi_ka-th)**2; dn2 = (th-lo_cs)**2 + (th-lo_ka)**2
lo_np, hi_np   = th - np.sqrt(dn2), th + np.sqrt(up2)
lo_all, hi_all = quad(None)

# ---------------------------------------------------------------- figure (Rivet style)
XMAX = 200.0
m = ctr < XMAX
E = np.concatenate([lo[m], [hi[m][-1]]])
fig, ax = plt.subplots(6, 1, figsize=(4.67, 9.36), sharex=True,
                       gridspec_kw={"height_ratios": [2.4, 1.25, 0.85, 0.85, 0.85, 0.85], "hspace": 0.0})
A = ax[0]
A.errorbar(ctr[m], dat[m], yerr=[dem[m], dep[m]], xerr=wid[m]/2, fmt="o", ms=2, lw=1,
           color="k", capsize=0, zorder=25, label="ATLAS Data, EPJC80(2020)616")
RED="#D62728"
A.stairs(hi_all[m], E, baseline=lo_all[m], fill=True, color=RED, alpha=0.20, lw=0, zorder=7)
A.stairs(th[m], E, color=RED, lw=1, zorder=7, label=r"N$^4$LL$^\prime$+N$^3$LO")
A.set_yscale("log"); A.set_ylabel(r"$1/\sigma\,$d$\sigma/$d$q_\mathrm{T}$ [1/GeV]", loc="top")
hh,ll=A.get_legend_handles_labels(); order=[ll.index("ATLAS Data, EPJC80(2020)616")]+[i for i,l in enumerate(ll) if not l.startswith("ATLAS")]
A.add_artist(A.legend([hh[i] for i in order],[ll[i] for i in order], alignment="left", loc="lower left", bbox_to_anchor=(0.01,0.01), markerfirst=True))
for k,txt in enumerate([r"$pp\to l^+l^-$, Born leptons", r"$p_{T,l}\ge 27$ GeV, $|\eta_l|\le 2.5$", r"$66$ GeV$\le m_{ll}\le 116$ GeV"]):
    A.text(0.04, 0.32+0.09*(2-k), txt, transform=A.transAxes, ha="left", va="bottom")
A.set_title("Transverse momentum of the lepton pair", loc="left")
B = ax[1]
B.axhline(1, color="k", lw=0.6, zorder=1)
B.stairs(hi_all[m]/dat[m], E, baseline=lo_all[m]/dat[m], fill=True, color=RED, alpha=0.20, lw=0, zorder=7)
B.stairs(th[m]/dat[m], E, color=RED, lw=1, zorder=7)
B.errorbar(ctr[m], np.ones(m.sum()), yerr=[dem[m]/dat[m], dep[m]/dat[m]], xerr=wid[m]/2, fmt="o",
           ms=2, lw=1, color="k", capsize=0, zorder=25)
B.set_ylabel("Theory~/~Data"); B.set_ylim(0.955, 1.045); B.yaxis.set_major_locator(MultipleLocator(0.02))
for a, (l, h, lab, col, hatch) in zip(ax[2:], [
        (lo_fo, hi_fo, "fixed order",             GREEN,  None),
        (lo_re, hi_re, "resummation",             ORANGE, None),
        (lo_np, hi_np, "Collins-Soper (lattice)", PURPLE, None),
        (th*(1-rel_num), th*(1+rel_num), "MC stat.", "0.35", "////")]):
    a.axhline(1, color="k", lw=0.6, zorder=1)
    if hatch: a.stairs(h[m]/th[m], E, baseline=l[m]/th[m], fill=False, edgecolor=col, lw=0, hatch=hatch)
    else:     a.stairs(h[m]/th[m], E, baseline=l[m]/th[m], fill=True, color=col, alpha=0.45, lw=0)
    a.set_ylim(0.955, 1.045); a.yaxis.set_major_locator(MultipleLocator(0.02))
    a.text(0.03, 0.10, lab, transform=a.transAxes, fontsize=8)
ax[3].set_ylabel("ratio to central", labelpad=6)
ax[-1].set_xlabel(r"$q_\mathrm{T}=p_\mathrm{T}^{\ell\ell}$ [GeV]", loc="right")
for a in ax: a.set_xscale("log")
ax[-1].set_xlim(1.0, XMAX)
plt.subplots_adjust(left=1.5*plt.rcParams["figure.subplot.left"], right=plt.rcParams["figure.subplot.right"], top=plt.rcParams["figure.subplot.top"], bottom=plt.rcParams["figure.subplot.bottom"])
fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"  wrote {OUT} (+ .png)")
r = th[m]/dat[m]; band = (hi_all[m]-lo_all[m])/2/th[m]
print(f"  theory/data: median |dev| = {100*np.median(np.abs(r-1)):.2f}%  max = {100*np.max(np.abs(r-1)):.2f}%")
print(f"  total theory band (median half-width) = {100*np.median(band):.2f}%")
