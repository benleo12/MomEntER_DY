"""Validate the SHIPPED per-event weights directly (no refit): histogram rT, acoplanarity and qT
with the exported weights and compare to Wan-Li's central prediction, normalised over the plotted
range, quoting the same trusted-region medians the pipeline prints (rT<2, qT<120, dphi all).

  ENERGY=13TeV_q30 QT_M=... python validate_shipped_weights.py [npz]
"""
import os, sys, re, json
import numpy as np, pandas as pd
E   = os.environ["ENERGY"]; MOM = f"moments_{E}"; PRIOR = f"sherpa_prior_{E}"
NPZ = sys.argv[1] if len(sys.argv) > 1 else f"{MOM}/sherpa_{E}_maxent_weights.npz"
# Same resolution chain as final_plots_pro: explicit env, a copy staged next to the pipeline, the
# author's laptop path.  A bare os.environ['QT_M'] made step [6/6] a KeyError on the cluster.
_qtag = E.split('_')[0]                      # 13TeV | 13p6TeV  (file tag)
_qdir = _qtag.replace('p','.') + '_IncPS'     # wju_13.6TeV_IncPS (directory)
_QT_NAME = f"qT_1D_Dist_NP_{_qtag}_IncPS_varmT.m"
QT_M = os.environ.get('QT_M') or next((p for p in (f"wju_{_qdir}/{_QT_NAME}",
        f"{os.path.dirname(os.path.abspath(__file__))}/wju_{_qdir}/{_QT_NAME}") if os.path.exists(p)), None)
if QT_M is None: sys.exit(f"validate: no qT theory file found for {E} (set QT_M=...)")
ACC = "N4LL'+N3LO"

import optimizer_DY_unc as o     # MUST use the pipeline's own loader: the npz event_index is
p = o.load_prior(PRIOR)          # defined against exactly this array (filtering included)
m, q, d, w0 = p["m"], p["pT"], p["d"], p["w"]
z = np.load(NPZ); w = z["weights"].astype(np.float64); idx = z["event_index"]
m, q, d, w0 = m[idx], q[idx], d[idx], w0[idx]
rT = q / m
print(f"  {E}: {len(w):,} shipped weights; sum w/sum w0 = {w.sum()/w0.sum():.4f}  "
      f"N_eff = {100*w.sum()**2/np.sum(w**2)/len(w):.2f}%  (prior N_eff = {100*w0.sum()**2/np.sum(w0**2)/len(w0):.2f}%)", flush=True)

def csv_dist(path, dist):
    t = pd.read_csv(path); t = t[(t.dist == dist) & (t.ScaleFO.str.startswith("CV")) & (t.ScaleRes.str.startswith("CV"))]
    e = np.concatenate([t.bin_lo.values, [t.bin_hi.values[-1]]]); return e, t.density.values
def m_dist(path, mat):
    txt = open(path, errors="ignore").read().replace("*^", "e")
    pat = mat + r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]\s*=\s*\{((?:\s*\{[^}]+\},?)+)\s*\}'
    nrm = lambda x: x.replace("'", "p").replace("+", "")
    for mm in re.finditer(pat, txt):
        if nrm(mm.group(1)) == nrm(ACC) and mm.group(2).startswith("CV") and mm.group(3).startswith("CV"):
            a = np.array([[float(x) for x in r.split(",")[:3]] for r in re.findall(r"\{([^{}]+)\}", mm.group(4))])
            return np.concatenate([a[:, 0], [a[-1, 1]]]), a[:, 2]
    raise SystemExit("central not found in " + path)

def report(tag, x, edges, thy, trust):
    unit = lambda h, e: (h / np.diff(e)) / np.sum(h / np.diff(e) * np.diff(e)) if h.sum() else h
    hr = unit(np.histogram(x, edges, weights=w)[0], edges)
    hp = unit(np.histogram(x, edges, weights=w0)[0], edges)
    th = thy / np.sum(thy * np.diff(edges))
    ctr = 0.5 * (edges[:-1] + edges[1:]); ok = (th > 0) & (hr > 0)
    tr = ok & (ctr < trust)
    dev = lambda h: 100 * np.abs(h[tr] / th[tr] - 1)
    print(f"  [{tag}] prior median {np.median(dev(hp)):.2f}%   REWEIGHTED trusted median {np.median(dev(hr)):.2f}%  "
          f"worst {np.max(dev(hr)):.2f}%   (bins {tr.sum()})", flush=True)

e, t = csv_dist(f"{MOM}/rTDist__N4LLp+N3LO.csv", "rTDist");   report("rT",   rT, e, t, 2.0)
e, t = csv_dist(f"{MOM}/dphiDist__N4LLp+N3LO.csv", "dphiDist"); report("dphi", d, e, t, 1e9)
e, t = m_dist(QT_M, "qTMat");                                  report("qT",   q,  e, t, 120.0)
