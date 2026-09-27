# Verification

This package is the code that produced the delivered products in `products/`.
Reproduction is checked at two levels by `verify.py`.

## Level 1 — self-contained (no prior needed)

Parses every delivered `lambda_export.json`, applies it to random events and to the
corners of phase space (zero and maximal acoplanarity, `q_T` far above the hand-off,
`m_ll` at the edge), and checks that every weight is finite, that the weight reverts
to the prior above the hand-off, that the multiplicity restriction holds on every
path and keeps the tail weight, and that bad inputs are refused.

```
$ python verify.py
[L1 13TeV       ] moments=15  schemes=29  finite=True  corners=True  restrict=True  bad_inputs_refused=True  reverts_above_200=True  K=0.8844 (29 schemes)
[L1 13p6TeV     ] moments=14  schemes=29  finite=True  corners=True  restrict=True  bad_inputs_refused=True  reverts_above_200=True  K=0.8833 (29 schemes)
[L1 13TeV_powheg] moments=19  schemes=29  finite=True  corners=True  restrict=True  bad_inputs_refused=True  ungated (pure reweighting everywhere)  K=0.9794 (29 schemes)
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

Result for v6.3 (commit 6945fb5, run on Perlmutter on 2026-09-27; the first 2 M events of each
prior, bulk region q_T < 120 GeV; no rescaling of either side):

| product      | events compared | median &#124;rel&#124; | p99 &#124;rel&#124; | max &#124;rel&#124; | K applied / expected | verdict |
|--------------|-----------------|------------|-----------|-----------|----------------------|---------|
| 13TeV        | 1,184,364       | 2.06e-08   | 5.28e-08  | 5.95e-08  | 0.884396 / 0.884396  | **PASS** |
| 13p6TeV      | 1,383,963       | 2.07e-08   | 5.29e-08  | 5.95e-08  | 0.883270 / 0.883270  | **PASS** |
| 13TeV_powheg | 1,985,484       | 2.61e-08   | 5.63e-08  | 5.96e-08  | 0.979378 / 0.979378  | **PASS** |

The weights re-derived from the delivered `lambda_export.json` alone equal the pipeline's
per-event weights to the precision of the stored multipliers, and the rate factor comes out
exactly where it is applied.  Level 1 passes on the same commit.  The apply-side changes of
v6.3 (see `CODE_MD5.txt`) therefore leave every delivered weight unchanged.

## Stored acoplanarity precision

The priors store Delta_phi with five to seven decimals, so the acoplanarity d = pi - Delta_phi
is quantized at about 1e-5 near d = 0.  Scanned on 2026-09-27 over every event: the Sherpa
priors have min d = 2.65e-6, no event with d <= 0, and 540 (13 TeV) or 483 (13.6 TeV) events
of 51.2 M with d <= 1e-5; the POWHEG prior has seven events of 48.6 M with d < 0 (down to
-3.5e-7, the rounding of an acos at Delta_phi = pi).  The moments were fitted on the stored
values, so the delivered multipliers are consistent with them.  At the smallest stored Sherpa
value, d = 2.65e-6 (q_T = 27 GeV, m_ll = 91 GeV), the weights are 1.16 (13 TeV), 0.70
(13.6 TeV) and 0.26 (POWHEG); the seven POWHEG events with d < 0 receive a weight of 1e-7 and
drop out.  `apply_lambdas` clips a negative acoplanarity to zero and floors the logarithm at
1e-12 where the pipeline floored it at 1e-30; the two differ only for d <= 0, i.e. for no
Sherpa event and for those seven POWHEG events.

## Full re-fit

The end-to-end fit (pools -> stability prune -> Newton solve -> export) is
deterministic and is reproduced by `run_pipeline.sh <ENERGY>`, given the prior
sample and the theory moments in `moments/`. The prior samples (tens of millions of
events, ~1-2 GB each) are too large for this repository and are available on
request.
