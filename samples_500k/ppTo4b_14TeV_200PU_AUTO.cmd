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
generate p p > b b b~ b~
output ppTo4b_14TeV_200PU_AUTO-v2
launch ppTo4b_14TeV_200PU_AUTO-v2
       shower=Pythia8
       detector=Delphes
       madspin=ON
       set iseed  # rnd seed 
       set ebeam1 7000.0 # beam 1 total energy in GeV
       set ebeam2 7000.0 # beam 2 total energy in GeV
       set nevents 
       set ptj 20
       set nhel 0
       set ickkw 1
       set ihtmin 300
       set xqcut 75.0
       ./Delphes/cards/CMS_PhaseII/CMS_PhaseII_200PU_v02.tcl
