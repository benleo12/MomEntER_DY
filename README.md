# MomEntER_DY

**Mom**ent **En**tropy **R**eweighting for **D**rell–**Y**an.

This repository is the code that produces the Drell–Yan particle-level results of
*"Lattice-Constrained Drell-Yan Resummation as Positive-Weight Events"*. It transfers
the accuracy of an N⁴LL′+N³LO resummed calculation — with a Collins–Soper kernel
fixed *ab initio* by lattice QCD — onto a standard Monte-Carlo event sample, as
**strictly positive, event-local weights**.

Given a prior sample with generator weights `w0`, the reweighted weight of each event is

```
w = w0 * exp( Σ_k λ_k g_k(v)  −  C )
```

where the `g_k` are moments of the two recoil observables — `rT = qT/m_ll` and the
acoplanarity `d = π − Δφ_ll` — and the multipliers `λ_k` are found by minimizing the
penalized maximum-entropy dual

```
L(λ) = log Z(λ)  −  Σ_k λ_k μ_k  +  ½ Σ_k σ_k² λ_k²
```

against the analytic target moments `μ_k` (with `σ_k` their theory uncertainties).
The weight depends only on the event, so it can be applied during generation. The
acoplanarity is the seed of the ATLAS φ\*_η observable (`φ*_η = tan(d/2) sin θ*_η`),
so matching `d` also predicts φ\*.

## Two ways to use it

### 1. Apply the delivered weights (no fitting)

The fitted multipliers for the published result ship in `products/`. Reweight your
own events in a few lines:

```python
from apply_lambdas import reweight
w = reweight(w0, qT, m_ll, dphi_ll, energy="13TeV")   # numpy arrays or scalars
```

You supply four per-event numbers: the generator weight `w0`, the dilepton `qT` and
`m_ll` [GeV], and the acoplanarity `dphi_ll = π − Δφ_ll`.
The 28 scale/NP variations are `reweight_scheme(..., scheme=s)` for `s in schemes()`;
the theory band combines the per-scale deviations in quadrature (pair the 0p5/2 variations of each scale). See `examples/apply_quickstart.py`.

**Hand-off (`gate`, default on).** By default the weight is smoothly returned to the
prior above a `qT` hand-off window, so a merged/matched prior keeps its own
multi-jet accuracy at large `qT` where the resummed input is neither valid nor
needed. This is the right choice for a prior you trust in the tail. Pass
`gate=False` to disable it and apply the pure reweighting across the full range
(appropriate when the prior is far from the calculation everywhere, or for a
diagnostic of the transfer itself):

```python
w = reweight(w0, qT, m_ll, dphi_ll, energy="13TeV", gate=False)   # no hand-off
```

The hand-off window is stored in the `gating` field of `lambda_export.json`.

**Generator scale variations.** If your prior carries its own scale-variation weights,
fit one multiplier set per variation (each on its own weight column) and take the
envelope over the resulting samples: below the hand-off the theory targets pin the
constrained spectra so the variations largely collapse, and what remains measures the
finite-moment-basis residual, while above the hand-off the envelope supplies the
generator uncertainty the reweighting itself cannot provide. Per variation V the event weight is the gated blend

    w_V = beta * [ w0_V * exp(sum_k lambda_V[k] phi_k - C_V) ]  +  (1 - beta) * w0_V

so below the hand-off (beta = 1) the varied prior is reweighted onto the theory target and
the seven samples collapse, while above it (beta = 0) the weights reduce to the bare prior
variation and the band there *is* the generator's own muR/muF uncertainty.

**Which weight family.** If your generator offers several scale-variation weight families,
use the **matrix-element-only** one: it is the counterpart of the fixed-order scales in the
calculation you are reweighting to. Families that also vary the parton shower move the prior
in a direction the theory targets have no counterpart for, and the reweighting is then asked
to undo a shower variation using constraints that cannot see it. For the shipped Sherpa prior
the two families differ on 3.15 percent of events (median ratio 1.6, some sign flips) and the
choice tightens the sub-hand-off band from 2.4 to 0.87 percent, leaving the tail unchanged at
about 17 percent. For POWHEG the question does not arise: `compute_rwgt` varies the matrix
element only, and all seven weights share one shower history.

**Generator tail variations (`w0_tail`).** Above the hand-off the events are the
generator's, so the uncertainty there should be the sample's own scale variation
(e.g. the standard 7-point muR/muF set). Pass each variation weight through the
tail branch and combine with the theory schemes (per-scale quadrature below the gate, generator variations above):

```python
band  = [reweight_scheme(w0, qT, m_ll, dphi_ll, scheme=s) for s in schemes()]   # theory, below gate
band += [reweight(w0, qT, m_ll, dphi_ll, w0_tail=w0_V) for w0_V in seven_point] # generator, above gate
```

The theory schemes revert to the central prior in the tail and the generator
variations act only there, so the two tile the phase space without double counting.

**Fit-sample recipe (default).** Statistical admission, stability pruning, and the
out-of-sample model selection run on a fixed 5M-event subsample plus the extreme
1e-5 tails of both observables. A moment is admitted only if this sample resolves
its target to the calculation's own precision, and only if what it adds beyond the
already-admitted moments is known better than a tenth of the prior's spread along
that new direction (the same bound is applied to its shift across the 28 scale
schemes). The delivered multipliers are then refit on the FULL sample
(`FULLFIT_EXPORT=1`, streaming), so the exported lambdas close the target moments
on all events.

### 2. Reproduce or re-fit from scratch

`run_pipeline.sh <ENERGY>` runs the full chain — candidate pools → stability prune →
Newton fit → out-of-sample model selection → export — for any prior:

```bash
./run_pipeline.sh 13TeV
```

This is the exact driver that produced `products/`. It needs a prior sample laid out
as `sherpa_prior_<ENERGY>/` (see [Input format](#input-format)) and the theory
moments in `moments_<ENERGY>/` (shipped here for 13TeV, 13p6TeV and 13TeV_powheg).

## What's in here

| Path | What it is |
|------|------------|
| `apply_lambdas.py` | event-local reweighter — apply the delivered weights (the product consumer) |
| `optimizer_DY_unc.py` | the MaxEnt engine: features, penalized dual, Newton solver with LM damping |
| `build_tau_sets.py` | candidate moment pools (precision screen, signal-to-noise) |
| `admit_prune.py` | statistical admission: only moments the prior can carry enter the fit |
| `select_stable.py` | automatic stability pruning to an absolute effective-event floor |
| `fit_health_v2.py`, `check_smallqt.py`, `health_shrink.py` | export health gate (small-qT divergence, N_eff floor, per-scheme dC) and the bounded shrink fallback |
| `validate_shipped_weights.py` | histograms the SHIPPED per-event weights against the calculation (no refit) |
| `final_plots_pro.py` | fit, uncertainty propagation, plots, and the `lambda_export.json` export |
| `run_pipeline.sh` | end-to-end driver (pools → prune → fit → select → export) |
| `verify.py` | reproduction checks (see [`VERIFY.md`](VERIFY.md)) |
| `moments_<E>/` | analytic N⁴LL′+N³LO moments and distributions, and the selected moment set, per energy |
| `products/<E>/` | **the delivered result**: `lambda_export.json` (+ 28-variation file, and for priors that carry generator scale weights a `lambda_export_prior_variations.json`) and the final plots |
| `examples/powheg_prior.md` | end-to-end recipe for reproducing the POWHEG+Pythia8 prior and its 7-point variations |

Three priors are shipped: `13TeV` and `13p6TeV` (Sherpa NLO multi-jet merged, the
published deliverables; 15 and 14 moments) and `13TeV_powheg` (19 moments, ungated):
a replica of the ATLAS MC15 sample DSID 361106 — POWHEG-BOX-V1 Z showered with
Pythia 8.186 (AZNLO tune, CTEQ6L1) and Photos++ QED FSR — the paper's
prior-independence demonstration on an independent generator. See
[`examples/powheg_prior.md`](examples/powheg_prior.md) for the generator
configurations and a template recipe for putting your own sample through the
pipeline. Its moment set differs from the Sherpa ones by design: the procedure is
prior-agnostic, the selection is prior-specific.

## Verification

`python verify.py` runs the self-contained checks. The delivered weights reproduce
the pipeline's per-event reference weights to machine precision (median relative
error ~2×10⁻⁸) at both energies — see [`VERIFY.md`](VERIFY.md).

## Install

```bash
pip install -r requirements.txt   # numpy, pandas, matplotlib. Python 3.9+.
```

## Input format

The reweighting needs, per event: `qT`, `m_ll`, the acoplanarity `d = π − Δφ_ll`, and
optionally a generator weight `w0`. For the pipeline (option 2) a prior directory
`sherpa_prior_<ENERGY>/` holds one gzipped CSV per quantity:

| File | Content |
|------|---------|
| `pT_values.csv.gz`  | dilepton `qT` [GeV] |
| `m_values.csv.gz`   | dilepton `m_ll` [GeV] |
| `dphi_values.csv.gz`| raw `Δφ_ll` (the loader forms `d = π − Δφ`) |
| `pT_weight.csv.gz`  | generator weight (optional; defaults to 1) |

The moments are computed for the inclusive phase space `m_ll > 40 GeV` with no lepton
cuts; fiducial cuts are applied downstream by the experiment. The prior samples used
for the published result are large (10⁷–10⁸ events) and are available on request.

## Citing

See [`CITATION.cff`](CITATION.cff). Please cite the paper.

## License

BSD-3-Clause — see [`LICENSE`](LICENSE).

## Selection and health guards

A moment enters the fit only if the prior can carry the constraint (`admit_prune.py`):
the sample must resolve its target to the calculation's own precision, and what it
adds beyond the already-admitted moments must be known better than 0.1 of the prior's
spread along that new direction, with the same bound applied to its shift across the
28 scale schemes. `select_stable.py` then prunes to a stable set; a fit that exhausts
its Newton budget (`MAX_STEPS=2000`) counts as unstable unless every moment already sits
within one standard deviation of its target. The Newton step is damped rather than
throttled: when a step would move one event's log-weight by more than the cap, the
Marquardt damping is raised and the step re-solved, so the step shortens along the
direction that makes it long and not along every direction at once (v6). Every export must pass the health gate (`fit_health_v2.py`): the weight
may not diverge as q_T → 0 (`check_smallqt.py`), the effective sample must stay above
an absolute floor, and no scale scheme may shift the normalization by more than
`|dC| < 0.5`. A failing member is dropped (`health_shrink.py`) and the fit is redone;
a fit that cannot pass is not shipped. A validation fit that does not converge is
treated like a non-converged export (v5.2): the moment with the largest stationarity
residual is dropped, the set is re-pruned and validated again, at most three times.
On the 2026-09-04 production samples, repeating the selection with other random fit
samples (seeds 7 and 2024 against the production seed 42) left the POWHEG set unchanged,
changed the 13 TeV set (seed 2024 kept 16 of the 17 production moments and added 4, K=20;
seed 7 kept 10 and added 2, K=12), and moved the agreement with the calculation by up to
about one percentage point. That study has not been repeated on the current samples. The out-of-sample selection additionally
refuses any moment set that agrees with the calculation *worse than the unreweighted
prior* on any validation distribution.

The rate factor `rate.K` sets the absolute scale, `sigma_calc / sigma_prior`. Every fitted
moment is normalised by the total rate, so `K` never enters the fit and changes no normalised
distribution. It is applied where the calculation is in control: `reweight()` multiplies the
reweighted branch and leaves the generator's tail above the hand-off alone (`rate_tail=True`
scales every event). `rate.sigma_calc_pb` and `rate.per_scheme[s].sigma_calc_pb` ship for every
setup; `K` is filled in wherever the prior's cross section is known: for POWHEG from the stored
per-event weights in pb, for the Sherpa setups from the trials column when the sample dumps
it (`sigma = sum(weights)/sum(trials)`, `rate.sigma_prior_source` says so); otherwise supply
it with `add_rate_block.py <tag> --sigma-prior <pb>`. Compare cross sections in the same
phase space first: `rate.phase_space` states it.

For a multi-jet merged prior the weight belongs to the 0-jet and 1-jet contributions
only: it is a function of q_T and the acoplanarity, so at higher multiplicity it would
erase what distinguishes a three-jet event from a one-jet event at the same q_T and
would replace the merged prediction's uncertainty, which grows with the number of jets,
by one from a calculation with at most three hard partons. Pass the multiplicity of the
hard process to `reweight(..., njet=...)` to restrict it; the multipliers themselves do
not depend on the multiplicity.

The delivered per-event weights are re-derived from `lambda_export.json` alone and
checked event-by-event against the pipeline's reference weights
(`validate_shipped_weights.py`, `verify.py --prior ...`); the release fingerprints in
`CODE_MD5.txt` and the relation to the production tree recorded in `PROVENANCE.diff`
tie this repository to the paper's runs.


## Paper figures

`bash figs/make_figures.sh` regenerates the paper figures from the shipped histogram inputs in `figs/data/` (see `figs/README.md`).
