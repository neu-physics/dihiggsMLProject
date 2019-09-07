set default_unset_couplings 99
set group_subprocesses Auto
set ignore_six_quark_processes False
set loop_optimized_output True
set loop_color_flows False
set gauge unitary
set complex_mass_scheme False
set max_npoint_for_channel 0
set nb_core 2
import model sm
define p = g u c d s u~ c~ d~ s~
define j = g u c d s u~ c~ d~ s~
define l+ = e+ mu+
define l- = e- mu-
define vl = ve vm vt
define vl~ = ve~ vm~ vt~
import model loop_sm
generate p p > h h [QCD]
output pp2hh4b_14TeV_0PU_AUTO-v2
launch pp2hh4b_14TeV_0PU_AUTO-v2
       shower=Pythia8
       detector=Delphes
       madspin=ON
       set iseed  # rnd seed 
       set ebeam1 7000.0 # beam 1 total energy in GeV
       set ebeam2 7000.0 # beam 2 total energy in GeV
       set nevents 
       set ptj 20
       set spinmode None
       decay h > b b~
       ./Delphes/cards/CMS_PhaseII/CMS_PhaseII_0PU_v02.tcl
