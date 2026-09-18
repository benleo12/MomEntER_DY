"""Agreement with the calculation as pulls: chi^2 per bin of the normalised spectrum against the
calculation, with the Monte-Carlo uncertainty of the calculation and of the sample added in
quadrature, over the bins below the hand-off (qT < 120 GeV, rT < 120 GeV/m_Z) and the full
acoplanarity range.  Also the fraction of bins within one sigma and within the calculation's
scale band.  Reads the H_<tag>.npz of make_paper_hists.py (needs the q_H2/r_H2/d_H2 arrays for
the sample's uncertainty; without them the calculation's uncertainty alone is used and said so).

    python pulls.py data/H_13TeV.npz [data/H_13p6TeV.npz ...]
"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import rivet_style as rs

HO = 120.0 / 91.19

def pulls(z, key):
    schs = [str(s) for s in z["schemes"]]
    e, H, T, tot = rs.thy_hists(z, key)
    have2 = f"{key}_H2" in z
    if have2:                                       # same rebinning as H (sums add)
        e0, H2 = z[f"{key}_e"], z[f"{key}_H2"]
        if key == "q": H2 = H2[: len(e) - 1]
        else: H2 = np.column_stack([rs.rebin(e0, H2[:, c], e, density=False) for c in range(H2.shape[1])])
    cal, st, sc = T["cen"], 0.5 * (T["sthi"] - T["stlo"]), 0.5 * (T["shi"] - T["slo"])
    c = 0.5 * (e[:-1] + e[1:]); w = np.diff(e)
    ok = (cal > 0) & np.isfinite(st) & (st > 0)
    if key == "q": ok &= c < 120
    if key == "r": ok &= c < HO
    out = {}
    for lab, col in (("prior", 0), ("rew", 1)):
        d = rs.dens(H[:, col], e, tot[col])
        ds = np.sqrt(H2[:, col]) / w / tot[col] if have2 else np.zeros_like(d)
        sig = np.sqrt(st ** 2 + ds ** 2)
        pull = (d[ok] - cal[ok]) / sig[ok]
        out[lab] = dict(chi2=float(np.mean(pull ** 2)), within1=float(np.mean(np.abs(pull) < 1)),
                        inscale=float(np.mean(np.abs(d[ok] - cal[ok]) <= sc[ok])), nbins=int(ok.sum()),
                        sample_err_med=float(np.median(ds[ok] / cal[ok])) if have2 else None,
                        calc_err_med=float(np.median(st[ok] / cal[ok])))
    return out, have2

if __name__ == "__main__":
    for f in sys.argv[1:]:
        z = np.load(f, allow_pickle=True)
        print(f"{f}: N_eff {100*float(z['neff']):.1f}%")
        for key, name in (("r", "rT"), ("d", "aco"), ("q", "qT")):
            o, have2 = pulls(z, key)
            s = f"  {name:3s} bins {o['rew']['nbins']:2d}  calc MC err {100*o['rew']['calc_err_med']:.2f}%"
            s += f"  sample err {100*o['rew']['sample_err_med']:.2f}%" if have2 else "  (sample err NOT included)"
            for lab in ("prior", "rew"):
                s += f" | {lab} chi2/bin {o[lab]['chi2']:5.1f} |pull|<1 {100*o[lab]['within1']:3.0f}% in scale band {100*o[lab]['inscale']:3.0f}%"
            print(s)
