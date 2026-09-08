#!/bin/bash
# Regenerate the paper figures from the shipped histogram inputs (no event samples needed).
#   bash figs/make_figures.sh            -> figs/out/
# Inputs (figs/data/): H_<setup>.npz  = prior | central | 28-scheme histograms of the reweighted samples
#   and the calculation (central, scale band, MC-stat band), written by make_paper_hists.py from the
#   event samples;  FIG_ATLASFID_powheg.npz = POWHEG fiducial histograms vs ATLAS (make_atlas_fid_pwg186p.py);
#   qT_fiducial_wunc_13TeV.m = the calculation in the ATLAS fiducial phase space (per-bin MC error);
#   ATLAS_2019_I1768911.yoda = the ATLAS measurement (HEPData / Rivet reference data).
# Style: default.mplstyle (the group's Rivet make-plots style; Palatino via LaTeX).
set -e; cd "$(dirname "$0")"; mkdir -p out
STACK=q,r,d python rivet_style.py thy data/H_13TeV.npz        out/sherpa_13TeV   "MEPS@NLO Prior" '$pp\to l^+l^-$, $\sqrt{s}=13$ TeV'
STACK=q,r,d python rivet_style.py thy data/H_13p6TeV.npz      out/sherpa_13p6TeV "MEPS@NLO Prior" '$pp\to l^+l^-$, $\sqrt{s}=13.6$ TeV'
ROW=1 STACK=q,d python rivet_style.py thy data/H_13TeV_powheg.npz out/powheg_13TeV "POWHEG Prior" '$pp\to l^+l^-$, $\sqrt{s}=13$ TeV'   # End Matter Fig. 3: panels side by side, spans both columns
ROW=1 python rivet_style.py fid data/FIG_ATLASFID_powheg.npz out/powheg_atlasfid                                                   # End Matter Fig. 4: side by side
python make_fig1_rivet.py out/fig_calc_vs_atlas.pdf                                                                                # Fig. 3
echo "figures written to figs/out/"
