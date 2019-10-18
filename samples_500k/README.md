# General Description
Directory with cards used to generate dihiggs -> 4b and QCD -> 4b samples using Madgraph_2.6.6 with Delpes_3.4.1 and the CMS_PhaseII_[0PU,200PU]_v02.tcl cards

Five samples:

1) dihiggs_CMS-PhaseII_0PU, dihiggs sample with 0 pileup
2) dihiggs_CMS-PhaseII_200PU, dihiggs sample with 200 pileup
3) QCD-HT300_CMS-PhaseII_0PU, qcd sample with generator level cut of HT > 300 GeV and 0 pileup
4) QCD-HT300_CMS-PhaseII_200PU, qcd sample with generator level cut of HT > 300 GeV and 200 pileup

# Madgraph Installation Instructions

0) Check if you have Python 2.6 or 2.7 installed on your machine. If you have a different version (i.e. Python 3.X), complete Steps A-D before moving to Step 1. If you do use system python of version 2.6 or 2.7, continue directly to Step 1

=====   Instructions for setting up python virtual env if using python version =/= 2.6 or 2.7   =====

A) Check /usr/bin/python* for the python executables available on your system. Hint, we want to use something like 2.7. If you don't have this, get it somehow

B) Create the virtual environment, e.g. $> virtualenv --python=/usr/bin/python2.7 <name of virtualenv>

C) Activate the environment, e.g. $> source <name of virtualenv>/bin/activate

======    Congratulations on your new virtual enviroment. You'll need to activate this every time you want to generate something in madgraph    ======

1) Get the most recent version of MG5 from https://launchpad.net/mg5amcnlo . Currently this is MG5_aMC_v2.6.6.tar.gz

2) Unpack tarball, e.g. $> tar -xzf MG5_aMC_v2.6.6.tar.gz

3) Move to MadGraph directory, e.g. $> cd MG5_aMC_v2_6_6/

4) Launch MadGraph, e.g. $> ./bin/mg5_aMC

5) Install pythia8 (this will take awhile) from mg command prompt, e.g. $> install pythia8

6) Install Delphes (this will take time but not as much as pythia) from mg command prompt, e.g. $> install Delphes

7) Now you need to install a bunch of other packages to generate the right stuff (this will take time) from mg command prompt, e.g. $> install collier ninja fastjet lhapdf5 iregi ExRootAnalysis

8) If you want to use the CMS_PhaseII detector in Delphes, you need to copy over a few .tcl files. First exit the MG5 interface and do the following
   $> cp Delphes/cards/CMS_PhaseII/muonMomentumResolution.tcl Templates/Common/Cards/
   $> cp Delphes/cards/CMS_PhaseII/muonTightId.tcl Templates/Common/Cards/
   $> cp Delphes/cards/CMS_PhaseII/muonLooseId.tcl Templates/Common/Cards/

9) You may need to edit the line in CMS_PhaseII_[0,200]PU_v02.tcl that specifies the location of the MinBias file. This is necessary even for 0PU for reasons


# Cross-sections for 500k samples from Madgraph
No analysis-level cuts. Errors are quadrature sums of all smaller production files, i.e. it's a very conservative over-estimation

## Dihiggs
Cross-section = 12.356281407 +/- 0.91194881995 fb 
Total Effective Lumi = 33034.1752419 pb

## QCD
Cross-section = 441866.0 +/- 30507.3550476 fb 
Total Effective Lumi = 1.35792436483 pb
