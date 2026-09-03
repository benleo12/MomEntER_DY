# Paper figures

`bash figs/make_figures.sh` regenerates every figure of the paper that comes from this pipeline,
from the histogram inputs in `figs/data/` (no event samples needed):

| output (figs/out/) | paper | content |
|---|---|---|
| `sherpa_13TeV_{q,r,d,stack}` | Results | Sherpa MEPS@NLO 13 TeV, prior / reweighted vs the calculation |
| `sherpa_13p6TeV_{q,r,d,stack}` | End Matter (prediction) | same at 13.6 TeV |
| `powheg_13TeV_stack` | End Matter Fig. 4 | POWHEG (ATLAS 361106 config), qT and acoplanarity vs the calculation |
| `powheg_atlasfid_stack` | End Matter Fig. 5 | POWHEG vs ATLAS data in the fiducial phase space (qT, phi*) |
| `fig_calc_vs_atlas` | Fig. 3 | the calculation vs ATLAS data with the uncertainty decomposition |

Scripts: `rivet_style.py` (plotter, Rivet make-plots layout and the group's `default.mplstyle`),
`make_fig1_rivet.py` (Fig. 3), `make_paper_hists.py` and `make_atlas_fid_pwg186p.py` (build the `.npz`
inputs from the event samples and the delivered lambdas; they need the samples and are documented in place).
Requires matplotlib with a LaTeX installation (Palatino); set `USETEX=0` to fall back to non-LaTeX text.
