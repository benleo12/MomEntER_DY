"""ATLAS-fiducial data comparison for the m>40 POWHEG (8.186+Photos) deliverable -- the End Matter
'data frame': q_T and phi*_eta in the ATLAS 1912.02844 fiducial volume (dressed leptons dR<0.1,
pT_l>27, |eta_l|<2.5, 66<m_ll<116), ATLAS d27/d28 (dressed, normalised), POWHEG+Pythia8 prior,
MaxEnt-reweighted POWHEG (inclusive m>40 lambdas applied per event, UNGATED) with the 29-scheme
band in per-knob quadrature.  House style of Fig. 1.

  python make_atlas_fid_pwg186p.py <csv_dir> <lambda_export.json> <lambda_export_variations.json> [out.pdf]
"""
import os, sys, glob, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"16")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from apply_lambdas import features

CSV, LAM, LAMV = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else "FIG_ATLASFID_pwg186p.pdf"
DAT = "atlas_data/ATLAS_2019_I1768911.yoda"

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.frameon": False, "figure.dpi": 160})

def load_atlas(path, hist):
    rows, inblk = [], False
    for line in open(path, errors="ignore"):
        if f"BEGIN YODA_SCATTER2D_V2 /REF/ATLAS_2019_I1768911/{hist}" in line: inblk = True; continue
        if inblk and line.startswith("END YODA"): break
        if inblk:
            p = line.split()
            if len(p) >= 6 and (p[0][0].isdigit() or p[0][0] == "-"):
                try: rows.append([float(x) for x in p[:6]])
                except ValueError: pass
    a = np.array(rows); x, exl, exh, y, eyl, eyh = a.T
    return np.concatenate([x-exl, [x[-1]+exh[-1]]]), y, 0.5*(eyl+eyh)

qt_edges, qt_dat, qt_err = load_atlas(DAT, "d27-x01-y01")
ps_edges, ps_dat, ps_err = load_atlas(DAT, "d28-x01-y01")
print(f"  ATLAS: qT {len(qt_dat)} bins [{qt_edges[0]:.0f},{qt_edges[-1]:.0f}]; phi* {len(ps_dat)} bins [{ps_edges[0]:.3f},{ps_edges[-1]:.2f}]", flush=True)

# ---- lambdas: central + schemes, one matrix ----
e = json.load(open(LAM)); v = json.load(open(LAMV))
names = e["moments"]; assert v["moments"] == names
schs = [s for s in v["schemes"] if s != "central"]
LAMM = np.column_stack([np.array(e["lambda_physical"], float)] + [np.array(v["schemes"][s]["lambda_physical"], float) for s in schs])
CC = np.array([float(e["log_norm_shift"])] + [float(v["schemes"][s]["log_norm_shift"]) for s in schs])
print(f"  lambdas: K={len(names)}, central + {len(schs)} schemes; gating applied = {e['gating'].get('applied', True)}", flush=True)

# ---- events: read only what the fiducial plot needs, keep only fiducial-passing events ----
cols = ["qT","m","dphi","phistar","ptl0","ptl1","etal0","etal1","w"]
Q, PS, W0, LG = [], [], [], []
ntot = 0
for i, f in enumerate(sorted(glob.glob(f"{CSV}/out-*.csv"))):
    d = pd.read_csv(f, usecols=cols); ntot += len(d)
    fid = (d.m > 66) & (d.m < 116) & (d.ptl0 > 27) & (d.ptl1 > 27) & (d.etal0.abs() < 2.5) & (d.etal1.abs() < 2.5)
    d = d[fid]
    Phi = features(names, d.qT.values / d.m.values, np.pi - d.dphi.values)     # acoplanarity = weight input
    LG.append((Phi @ LAMM).astype(np.float64)); Q.append(d.qT.values); PS.append(d.phistar.values); W0.append(d.w.values)
    if i % 16 == 0: print(f"    {i}/128 files, fiducial so far {sum(len(q) for q in Q):,}", flush=True)
qT = np.concatenate(Q); phistar = np.concatenate(PS); w0 = np.concatenate(W0); LG = np.concatenate(LG)
print(f"  POWHEG: {ntot:,} showered, {len(qT):,} pass ATLAS fiducial ({100*len(qT)/ntot:.1f}%)", flush=True)
WS = w0[:, None] * np.exp(LG - CC[None, :])          # column 0 = central, then schemes; UNGATED
wrew = WS[:, 0]
print(f"  central: sum w_rew/sum w0 = {wrew.sum()/w0.sum():.4f}  N_eff = {100*wrew.sum()**2/np.sum(wrew**2)/len(wrew):.2f}%", flush=True)

def norm_hist(x, ww, edges):
    # DENSITY: counts / bin width / total  (the ATLAS d27/d28 tables are densities, sum(y*w)=1)
    h, _ = np.histogram(x, bins=edges, weights=ww)
    return h / np.diff(edges) / h.sum()
def _knob(lbl):
    return lbl[3:] if lbl.startswith("0p5") else (lbl[1:] if lbl.startswith("2") else lbl)
def band(x, edges, rew):
    kn = {}
    for j, s in enumerate(schs): kn.setdefault(_knob(s), []).append(norm_hist(x, WS[:, j+1], edges))
    up2, dn2 = np.zeros_like(rew), np.zeros_like(rew)
    for vs in kn.values():
        up2 += np.maximum.reduce([np.maximum(v - rew, 0) for v in vs])**2
        dn2 += np.maximum.reduce([np.maximum(rew - v, 0) for v in vs])**2
    return rew - np.sqrt(dn2), rew + np.sqrt(up2)

def panel(ax, edges, x, dat, err, xlabel, xmax, logx=False):
    pri = norm_hist(x, w0, edges); rew = norm_hist(x, wrew, edges); rlo, rhi = band(x, edges, rew)
    ctr = 0.5*(edges[:-1]+edges[1:]); wid = np.diff(edges); m_ = ctr < xmax
    ee = np.concatenate([edges[:-1][m_], [edges[1:][m_][-1]]])
    ax.axhline(1, color="k", lw=0.7)
    ax.stairs(rhi[m_]/dat[m_], ee, baseline=rlo[m_]/dat[m_], fill=True, color="C0", alpha=0.25, lw=0, label="scale variations")
    ax.stairs(pri[m_]/dat[m_], ee, color="0.45", lw=1.3, ls="--", label="POWHEG+Pythia8 (AZNLO)")
    ax.stairs(rew[m_]/dat[m_], ee, color="C0", lw=1.5, label="MaxEnt reweighted")
    ax.errorbar(ctr[m_], np.ones(m_.sum()), yerr=err[m_]/dat[m_], xerr=wid[m_]/2, fmt="o", ms=2.2, lw=0.8, color="k", capsize=0, zorder=5, label="ATLAS")
    ax.set_ylabel("ratio to data"); ax.set_ylim(0.80, 1.20)
    ax.yaxis.set_major_locator(MultipleLocator(0.1)); ax.set_xlabel(xlabel)
    if logx: ax.set_xscale("log"); ax.set_xlim(max(edges[0], 1.0), xmax)
    else:    ax.set_xlim(edges[0], xmax); ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return pri, rew, rlo, rhi

fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.4), gridspec_kw={"hspace": 0.32})
res = {}
res["qT"] = panel(ax[0], qt_edges, qT, qt_dat, qt_err, r"$q_T = p_T^{\ell\ell}$  [GeV]", 200.0)
res["phi*"] = panel(ax[1], ps_edges, phistar, ps_dat, ps_err, r"$\phi^*_\eta$", float(ps_edges[-1]))
ax[0].legend(loc="lower left", fontsize=6.5, ncol=1)
ax[0].text(0.97, 0.06, r"$pp\to\ell^+\ell^-$, $\sqrt{s}=13$ TeV, ATLAS fiducial", transform=ax[0].transAxes, ha="right", fontsize=6.8)
np.savez(OUT.replace(".pdf", "_hists.npz"), qt_edges=qt_edges, qt_dat=qt_dat, qt_err=qt_err, qt_pri=res["qT"][0], qt_rew=res["qT"][1], qt_rlo=res["qT"][2], qt_rhi=res["qT"][3],
         ps_edges=ps_edges, ps_dat=ps_dat, ps_err=ps_err, ps_pri=res["phi*"][0], ps_rew=res["phi*"][1], ps_rlo=res["phi*"][2], ps_rhi=res["phi*"][3])
fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"  wrote {OUT} (+ .png)")
for tag, edges, dat, err in [("qT", qt_edges, qt_dat, qt_err), ("phi*", ps_edges, ps_dat, ps_err)]:
    pri, rew, rlo, rhi = res[tag]; ctr = 0.5*(edges[:-1]+edges[1:]); mm = ctr < (200 if tag == "qT" else edges[-1])
    print(f"\n  {tag}: prior/data |dev| med={100*np.median(np.abs(pri[mm]/dat[mm]-1)):.2f}%  reweighted/data med={100*np.median(np.abs(rew[mm]/dat[mm]-1)):.2f}%  worst={100*np.max(np.abs(rew[mm]/dat[mm]-1)):.2f}%")
    print(f"  {'bin':>14} {'prior/data':>10} {'rew/data':>9} {'band':>13} {'data unc %':>10}")
    for j in np.where(mm)[0]:
        print(f"  {edges[j]:7.3g}-{edges[j+1]:<6.3g} {pri[j]/dat[j]:10.3f} {rew[j]/dat[j]:9.3f} [{rlo[j]/dat[j]:5.3f},{rhi[j]/dat[j]:5.3f}] {100*err[j]/dat[j]:10.2f}")
