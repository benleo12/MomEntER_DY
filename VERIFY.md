# Verification

This package is the code that produced the delivered products in `products/`.
Reproduction is checked at two levels by `verify.py`.

## Level 1 — self-contained (no prior needed)

Parses every delivered `lambda_export.json`, applies it to random events, and
checks the weights are finite, that the normalization constant is present, and
that the weight reverts to the prior above the `q_T` hand-off.

```
$ python verify.py
[L1 13TeV   ] moments=41  schemes=29  finite=True  reverts_above_200=True
[L1 13p6TeV ] moments=28  schemes=29  finite=True  reverts_above_200=True
L1: PASS
```

## Level 2 — full reproduction against the reference weights

The pipeline (`final_plots_pro.py`, `EXPORT`/`WRITE_WEIGHTS` modes) writes both the
per-moment multipliers `lambda_export.json` and the per-event weights it implies.
Level 2 re-derives the per-event weights **from the delivered `lambda_export.json`
alone**, using the event-local `apply_lambdas.reweight`, and compares them to the
pipeline's reference weights on the same prior.

```
$ python verify.py --prior /path/to/sherpa_prior_13TeV --ref sherpa_13TeV_maxent_weights.npz --energy 13TeV
```

Result at release (2 M events per energy, bulk region q_T < 120 GeV):

| energy   | median &#124;rel&#124; | p99 &#124;rel&#124; | ratio  | verdict |
|----------|------------|-----------|--------|---------|
| 13 TeV   | 2.07e-08   | 5.29e-08  | 1.000000 | **reproduces** |
| 13.6 TeV | 2.07e-08   | 5.28e-08  | 1.000000 | **reproduces** |

The delivered `lambda_export.json` files are byte-for-byte identical to those sent
to the collaboration, so the applied weights agree with the reference to machine
precision (~2e-8): the package reproduces the delivered result exactly.

## Full re-fit

The end-to-end fit (pools -> stability prune -> Newton solve -> export) is
deterministic and is reproduced by `run_pipeline.sh <ENERGY>`, given the prior
sample and the theory moments in `moments/`. The prior samples (tens of millions of
events, ~1-2 GB each) are too large for this repository and are available on
request.
