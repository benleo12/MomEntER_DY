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

**Scale nuisance sets.** `products/<E>/lambda_export_nuisance.json` compresses the
28 schemes into 3 orthogonal nuisance directions (99+ percent of the scale
variance), each shipped as +-1 sigma multiplier sets, analogous to Hessian PDF
error sets. Coverage is still quoted from the per-scale quadrature of the schemes, the nuisance sets
provide the correlated parametrization downstream fits need. Rebuild with
`make_nuisance_sets.py`.

### 2. Reproduce or re-fit from scratch

`run_pipeline.sh <ENERGY>` runs the full chain — candidate pools → stability prune →
Newton fit → out-of-sample model selection → export — for any prior:

```bash
./run_pipeline.sh 13TeV
```

This is the exact driver that produced `products/`. It needs a prior sample laid out
as `sherpa_prior_<ENERGY>/` (see [Input format](#input-format)) and the theory
moments in `moments/`.

## What's in here

| Path | What it is |
|------|------------|
| `apply_lambdas.py` | event-local reweighter — apply the delivered weights (the product consumer) |
| `optimizer_DY_unc.py` | the MaxEnt engine: features, penalized dual, Newton solver with LM damping |
| `build_tau_sets.py` | candidate moment pools (precision screen, signal-to-noise) |
| `select_stable.py` | automatic stability pruning to an absolute effective-event floor |
| `final_plots_pro.py` | fit, uncertainty propagation, plots, and the `lambda_export.json` export |
| `run_pipeline.sh` | end-to-end driver (pools → prune → fit → select → export) |
| `verify.py` | reproduction checks (see [`VERIFY.md`](VERIFY.md)) |
| `moments/<E>/` | analytic N⁴LL′+N³LO moments and distributions, and the selected moment set, per energy |
| `products/<E>/` | **the delivered result**: `lambda_export.json` (+ 28-variation file) and the final plots |

Two energies are shipped: `13TeV` (41 moments) and `13p6TeV` (28 moments).

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
