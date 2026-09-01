# Building a POWHEG+Pythia8 prior (13 TeV)

> **Shipped configuration.** `products/13TeV_powheg/` was produced from a replica of
> the ATLAS MC15 sample DSID 361106: POWHEG-BOX-**V1** Z
> ([`powheg_input_361106`](powheg_input_361106), `mass_low = 35` GeV) showered with
> **Pythia 8.186** using the AZNLO tune + CTEQ6L1 and Photos++ for QED FSR
> ([`pythia_aznlo_8186_photos.cmnd`](pythia_aznlo_8186_photos.cmnd)), dressed
> leptons (ΔR < 0.1), analysed for m_ll > 40 GeV. It validates against the ATLAS
> sample at the 0.1% level where the two overlap.

The recipe below builds a comparable sample with the current POWHEG-BOX-V2 and
shower-veto matching, carrying 7-point matrix-element scale variations — use it as a
template for putting your own generator through the same pipeline.

Everything below was used to produce the shipped numbers. All the scripts referenced
live in this `examples/` directory.

## 1. Generate the NLO events (POWHEG-BOX-V2, Z process)

Build the `Z` process of POWHEG-BOX-V2, then run `pwhg_main` with `powheg_input`
(copied here as `powheg_input_Z13TeV`). The settings that matter:

    ebeam1/2  6500d0          13 TeV
    lhans1/2  14400           CT18NLO
    vdecaymode 1              Z -> e+ e-
    mass_low  35              generation window, wider than the m_ll > 40 analysis cut
    mass_high 2000
    numevts   1150000

This writes `pwgevents.lhe` (~1.2 GB for 1.15M events). POWHEG stores per-event
`#rwgt` records, which is what makes step 2 possible without regenerating.

## 2. Add the 7-point matrix-element scale weights

The scale variations are obtained by *reweighting the same events*, so every variation
shares the event sample and only the weight changes. Insert an `<initrwgt>` block in
the LHE header (POWHEG needs it to attach weights), then run one `compute_rwgt` pass
per variation with `powheg_rwgt_chain.sh`:

    (muR, muF) = (1/2,1/2), (1/2,1), (1,1/2), (1,2), (2,1), (2,2)

Each pass appends one weight per event. POWHEG writes them in its old
`#new weight,...` format, so convert them into LHEF3 `<rwgt>` blocks that Pythia
parses natively:

    python powheg_to_lhef3.py pwgevents.lhe pwgevents-lhef3.lhe

Six passes plus the conversion take about 20 minutes for 1.15M events: reweighting is
cheap because the kinematics are fixed and only the weight is recomputed.

## 3. Shower with Pythia 8 (standard POWHEG matching)

`powheg_shower_hooks.cc` showers the LHE and writes the per-event CSV the pipeline
consumes. It uses the standard matching prescription, not a hand-rolled scale choice:

    POWHEG:veto = 1, POWHEG:nFinal = -1, POWHEG:pThard = 0,
    POWHEG:pTemt = 0, POWHEG:pTdef = 1, POWHEG:vetoCount = 3, POWHEG:QEDveto = 2
    SpaceShower:pTmaxMatch = 2, TimeShower:pTmaxMatch = 2
    PowhegHooks attached as the user hook
    Tune:pp = 14 (Monash 2013), PartonLevel:MPI = on

Leptons are dressed with FSR photons within dR < 0.1 and the analysis cut m_ll > 40 GeV
is applied. Build and run:

    g++ -O2 -o powheg_shower_hooks powheg_shower_hooks.cc $(pythia8-config --cxxflags --libs)
    ./powheg_shower_hooks pwgevents-lhef3.lhe 1150000 hooks_wgt.csv

Columns: `qT,m,dphi,phistar,ptl0,ptl1,etal0,etal1,w,w1002..w1007`, where `w` is the
central weight and `w1002..w1007` are the six scale variations of the same event.

## 4. Lay out the prior directory

    sherpa_prior_13TeV_powheg/
      pT_values.csv.gz     <- qT column
      m_values.csv.gz      <- m column
      dphi_values.csv.gz   <- dphi column (raw Delta-phi; the loader forms pi - Delta-phi)
      pT_weight.csv.gz     <- w column
      variations/MUR_0.5__MUF_0.5.csv.gz ... MUR_2__MUF_2.csv.gz   <- w1002..w1007

`powheg_build_prior.py` does this directly from `hooks_wgt.csv`.

## 5. Run the pipeline

    ./run_pipeline.sh 13TeV_powheg

Selection screens and pruning run on a 2M-event subsample; the delivered multipliers
are refit on the full sample. On this prior the procedure keeps 61 moments
(winner pool: tau2.0) and reproduces the calculation to 1.1% in rT, 2.2% in
the acoplanarity and 1.7% in qT, out of sample.

To also get the shower-scale band, fit one multiplier set per variation (each on its
own weight column) and take the envelope over the seven resulting samples. The
validation plotter picks this up automatically when
`products/<energy>/lambda_export_prior_variations.json` and the `variations/`
directory are present, and draws it as the "Shower unc." band.

## Using the shipped result instead

If you only want the delivered weights, skip everything above:

    from apply_lambdas import reweight
    w = reweight(w0, qT, m_ll, dphi_ll, jpath="products/13TeV_powheg/lambda_export.json")

The selected moment set for this prior differs from the Sherpa ones. That is the
method working as intended: the *procedure* is prior-agnostic, the *selection* is
prior-specific, because which moments can be imposed stably depends on the weight
structure and tails of the sample being reweighted.
