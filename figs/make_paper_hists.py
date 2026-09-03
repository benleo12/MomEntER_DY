"""Paper histograms for one prior tag: prior, MaxEnt central and the 28 scheme columns, binned on
  q_T  : ATLAS 1912.02844 d27 edges (full range) -- theory N4LL'+N3LO mapped onto them below 200 GeV
  r_T  : Wan-Li rTDist edges (moments dir CSV)
  d    : acoplanarity pi-dphi, Wan-Li dphiDist edges
plus the theory central / per-knob-quadrature scale band / MC-stat band on the same edges.
Weights are the SHIPPED ones: w = b(qT) w0 exp(lambda.phi - C) + (1-b) w0 with the export's gate window
(POWHEG: window [1e9,2e9] => ungated).  The central column is cross-checked event by event against
the exported *_maxent_weights.npz.  Everything is saved raw (no coarsening, no band) so the plotter
can rebin and band consistently.

  TAG=13TeV_q30 python make_paper_hists.py [out.npz]        env: NEV (0=all) CHUNK
"""
import os, sys, re, json, glob
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "16")
import numpy as np, pandas as pd
from apply_lambdas import features

TAG   = os.environ["TAG"]
OUT   = sys.argv[1] if len(sys.argv) > 1 else f"PAPER_HISTS_{TAG}.npz"
ACC   = "N4LL'+N3LO"
PRIOR = os.environ.get("PRIOR_DIR", f"sherpa_prior_{TAG}")
MOM   = os.environ.get("MOM_DIR",   f"moments_{TAG}")
ETAG  = "13p6TeV" if TAG.startswith("13p6") else "13TeV"
WJU   = os.environ.get("WJU_DIR", f"wju_{'13.6TeV' if ETAG == '13p6TeV' else '13TeV'}_IncPS")
QT_M  = f"{WJU}/qT_1D_Dist_NP_{ETAG}_IncPS_varmT.m"
NEV   = int(os.environ.get("NEV", "0"))
CH    = int(os.environ.get("CHUNK", "4000000"))
QTH   = 200.0                                             # edge of the N4LL'+N3LO qT prediction

# ---------------- theory ------------------------------------------------------------------
def _knob(lbl): return lbl[3:] if lbl.startswith("0p5") else (lbl[1:] if lbl.startswith("2") else lbl)
def _lbl(fo, res):
    fo, res = str(fo), str(res)
    return fo if res.startswith("CV") and not fo.startswith("CV") else res

def band_from(edges, cn, var):
    """var: {label: density}; per-knob envelope, knobs in quadrature (make_fig1 / End Matter convention)."""
    kn = {}
    for lbl, v in var.items(): kn.setdefault(_knob(lbl.split("->")[0]), []).append(v)
    up2, dn2 = np.zeros_like(cn), np.zeros_like(cn)
    for vs in kn.values():
        up2 += np.maximum.reduce([np.maximum(v - cn, 0) for v in vs])**2
        dn2 += np.maximum.reduce([np.maximum(cn - v, 0) for v in vs])**2
    return cn - np.sqrt(dn2), cn + np.sqrt(up2), sorted(kn)

def unit(edges, dens): return dens / np.sum(dens * np.diff(edges))

def theory_csv(path):
    df = pd.read_csv(path); df = df[df.acc.str.replace("'", "p").str.replace("+", "") == ACC.replace("'", "p").replace("+", "")]
    grp = {(a, b): g.sort_values("bin_lo") for (a, b), g in df.groupby(["ScaleFO", "ScaleRes"])}
    cen = [k for k in grp if str(k[0]).startswith("CV") and str(k[1]).startswith("CV")]; assert len(cen) == 1, cen
    g = grp[cen[0]]; e = np.concatenate([g.bin_lo.values, [g.bin_hi.values[-1]]])
    Z = np.sum(g.density.values * np.diff(e)); c = g.density.values / Z; s = g.uncertainty.values / Z
    var = {_lbl(a, b): unit(e, grp[(a, b)].density.values) for (a, b) in grp if (a, b) != cen[0]}
    lo, hi, knobs = band_from(e, c, var)
    return dict(e=e, cen=c, slo=lo, shi=hi, stlo=c - s, sthi=c + s), knobs

def load_m(path, mat):
    txt = open(path, errors="ignore").read().replace("*^", "e")
    pat = (mat + r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]\s*=\s*\{((?:\s*\{[^}]+\},?)+)\s*\}')
    def norm(x): return x.replace("'", "p").replace("+", "")
    out = {}
    for m in re.finditer(pat, txt):
        if norm(m.group(1)) != norm(ACC): continue
        arr = []
        for r in re.findall(r"\{([^{}]+)\}", m.group(4)):
            f = [x.strip() for x in r.split(",")]
            if len(f) < 3: continue
            try: arr.append([float(f[0]), float(f[1]), float(f[2]), float(f[3]) if len(f) >= 4 else 0.0])
            except ValueError: pass
        if arr: out[(m.group(2), m.group(3))] = np.array(arr)
    cen = [k for k in out if k[0].startswith("CV") and k[1].startswith("CV")]; assert cen, f"central not found in {path}"
    c = out[cen[0]]; e = np.concatenate([c[:, 0], [c[-1, 1]]])
    return e, c[:, 2], c[:, 3], {_lbl(k[1], k[0]): v[:, 2] for k, v in out.items() if k != cen[0]}

def theory_m(path, mat):
    e, c, cunc, var = load_m(path, mat)
    Z = np.sum(c * np.diff(e)); cn = c / Z; sn = cunc / Z
    lo, hi, knobs = band_from(e, cn, {k: unit(e, v) for k, v in var.items()})
    return dict(e=e, cen=cn, slo=lo, shi=hi, stlo=cn - sn, sthi=cn + sn), knobs

def atlas(hist):
    rows, inb = [], False
    for line in open("atlas_data/ATLAS_2019_I1768911.yoda", errors="ignore"):
        if f"BEGIN YODA_SCATTER2D_V2 /REF/ATLAS_2019_I1768911/{hist}" in line: inb = True; continue
        if inb and line.startswith("END YODA"): break
        if inb:
            p = line.split()
            if len(p) >= 6 and (p[0][0].isdigit() or p[0][0] == "-"):
                try: rows.append([float(x) for x in p[:6]])
                except ValueError: pass
    a = np.array(rows); x, exl, exh, y, eyl, eyh = a.T
    return np.concatenate([x - exl, [x[-1] + exh[-1]]]), y, 0.5 * (eyl + eyh)

def onto(es, dens, ed):
    """average a density given on edges es onto edges ed; NaN where ed is not covered by es"""
    out = np.full(len(ed) - 1, np.nan)
    for j in range(len(out)):
        if ed[j] < es[0] - 1e-9 or ed[j + 1] > es[-1] + 1e-9: continue
        l = np.maximum(ed[j], es[:-1]); h = np.minimum(ed[j + 1], es[1:])
        out[j] = np.sum(dens * np.clip(h - l, 0, None)) / (ed[j + 1] - ed[j])
    return out

TQ, kq = theory_m(QT_M, "qTMat")
TR, kr = theory_csv(f"{MOM}/rTDist__N4LLp+N3LO.csv")
TD, kd = theory_csv(f"{MOM}/dphiDist__N4LLp+N3LO.csv")
AE, ADAT, AERR = atlas("d27-x01-y01")                     # full ATLAS qT binning (inclusive theory only below 200)
TQa = {k: onto(TQ["e"], v, AE) for k, v in TQ.items() if k != "e"}; TQa["e"] = AE
print(f"  theory qT: {len(TQ['cen'])} bins [{TQ['e'][0]:.0f},{TQ['e'][-1]:.0f}] -> ATLAS {len(AE)-1} bins [{AE[0]:.0f},{AE[-1]:.0f}], knobs {kq}", flush=True)
print(f"  theory rT: {len(TR['cen'])} bins [{TR['e'][0]:.2f},{TR['e'][-1]:.2f}] knobs {kr};  d: {len(TD['cen'])} bins [{TD['e'][0]:.3f},{TD['e'][-1]:.3f}] knobs {kd}", flush=True)

# ---------------- lambdas + gate ------------------------------------------------------------
e_ = json.load(open(f"{MOM}/lambda_export.json")); v_ = json.load(open(f"{MOM}/lambda_export_variations.json"))
names = e_["moments"]; assert v_["moments"] == names
schs = [s for s in v_["schemes"] if s != "central"]
LAMM = np.column_stack([np.array(e_["lambda_physical"], float)] + [np.array(v_["schemes"][s]["lambda_physical"], float) for s in schs])
CC   = np.array([float(e_["log_norm_shift"])] + [float(v_["schemes"][s]["log_norm_shift"]) for s in schs])
gat  = e_["gating"]; GLO, GHI = [float(x) for x in gat["window_GeV"]]; GATED = gat.get("applied", True) and GHI < 1e8
print(f"  lambdas: K={len(names)}, central + {len(schs)} schemes; gate window [{GLO:g},{GHI:g}] applied={GATED}", flush=True)

# ---------------- prior (identical filtering to optimizer_DY_unc.load_prior) ---------------
rd = lambda f: pd.to_numeric(pd.read_csv(f"{PRIOR}/{f}", header=None, low_memory=False).iloc[:, 0], errors="coerce").values.astype(np.float64)
D, Q, M = rd("dphi_values.csv.gz"), rd("pT_values.csv.gz"), rd("m_values.csv.gz")
n = min(map(len, (M, Q, D))); D, Q, M = D[:n], Q[:n], M[:n]
try: W0 = rd("pT_weight.csv.gz")[:n]
except Exception: W0 = np.ones(n)
ok = np.isfinite(M) & (M > 1e-300) & np.isfinite(D) & np.isfinite(Q) & np.isfinite(W0)
D, Q, M, W0 = D[ok], Q[ok], M[ok], W0[ok]
if NEV: D, Q, M, W0 = D[:NEV], Q[:NEV], M[:NEV], W0[:NEV]
N = len(M)
print(f"  prior {PRIOR}: {n:,} rows, {N:,} good; negative weights {100*np.mean(W0<0):.2f}%; m<40: {100*np.mean(M<40):.3f}%", flush=True)

wexp = None
try:
    z = np.load(glob.glob(f"{MOM}/*_maxent_weights.npz")[0]); wexp = np.full(N, np.nan); wexp[z["event_index"]] = z["weights"]
    print(f"  export weights: {len(z['weights']):,} events (index max {z['event_index'].max():,})", flush=True)
except Exception as ex: print(f"  export weights not compared: {ex}")

ED = {"q": AE, "r": TR["e"], "d": TD["e"]}
H  = {k: np.zeros((len(e) - 1, LAMM.shape[1] + 1)) for k, e in ED.items()}    # column 0 = prior, 1 = central, 2.. schemes
s1 = s2 = 0.0; maxdiff = 0.0; over = np.zeros(3)                                  # events above each axis' last edge
for a in range(0, N, CH):
    b = min(a + CH, N)
    Phi = features(names, Q[a:b] / M[a:b], np.pi - D[a:b])
    WR  = W0[a:b, None] * np.exp((Phi @ LAMM) - CC[None, :])
    if GATED:
        t = np.clip((Q[a:b] - GLO) / (GHI - GLO), 0, 1); beta = 1.0 - (6*t**5 - 15*t**4 + 10*t**3)
        WR = beta[:, None] * WR + (1.0 - beta)[:, None] * W0[a:b, None]
    WS = np.column_stack([W0[a:b], WR])
    if wexp is not None:
        m_ = np.isfinite(wexp[a:b]); maxdiff = max(maxdiff, float(np.max(np.abs(WR[m_, 0] - wexp[a:b][m_]) / np.maximum(np.abs(wexp[a:b][m_]), 1e-300))) if m_.any() else 0.0)
    for k, x in (("q", Q[a:b]), ("r", Q[a:b] / M[a:b]), ("d", np.pi - D[a:b])):
        e = ED[k]; i = np.searchsorted(e, x, side="right") - 1; okb = (i >= 0) & (i < len(e) - 1)
        for c in range(WS.shape[1]): H[k][:, c] += np.bincount(i[okb], weights=WS[okb, c], minlength=len(e) - 1)
    w = WR[:, 0]; s1 += w.sum(); s2 += np.sum(w**2)
    print(f"    {b:,}/{N:,}", flush=True)
neff = s1**2 / s2 / N
print(f"  central: sum w/sum w0 = {s1/W0.sum():.4f}   N_eff = {100*neff:.2f}%   max rel. diff vs exported weights = {maxdiff:.2e}", flush=True)

def dens(h, e): d_ = h / np.diff(e); return d_ / np.sum(d_ * np.diff(e))
for k, T, lim in (("q", TQa, QTH), ("r", TR, 1e9), ("d", TD, 1e9)):
    e = ED[k]; ctr = 0.5 * (e[:-1] + e[1:]); m_ = (ctr < lim) & np.isfinite(T["cen"]) & (T["cen"] > 0)
    p_ = dens(H[k][:, 0], e)[m_] / T["cen"][m_]; r_ = dens(H[k][:, 1], e)[m_] / T["cen"][m_]
    print(f"  {k}: prior/theory median |dev| {100*np.median(np.abs(p_-1)):.2f}% (max {100*np.max(np.abs(p_-1)):.1f}%)  reweighted {100*np.median(np.abs(r_-1)):.2f}% (max {100*np.max(np.abs(r_-1)):.1f}%)", flush=True)

np.savez(OUT, tag=TAG, names=np.array(names), schemes=np.array(schs), N=N, neff=neff, gated=GATED, gate_window=np.array([GLO, GHI]),
         q_e=AE, q_H=H["q"], q_dat=ADAT, q_err=AERR, **{f"q_{k}": v for k, v in TQa.items() if k != "e"},
         r_e=TR["e"], r_H=H["r"], **{f"r_{k}": v for k, v in TR.items() if k != "e"},
         d_e=TD["e"], d_H=H["d"], **{f"d_{k}": v for k, v in TD.items() if k != "e"})
print(f"  wrote {OUT}", flush=True)
