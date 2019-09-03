source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc62-opt/setup.csh
cp ../MG5_aMC_v2.6.6.tar.gz .
tar -xvzf MG5_aMC_v2.6.6.tar.gz
cd MG5_aMC_v2_6_6

cp ../installPackages*.cmd . #some line to wget install script or just pass via condor submission
cp ../test_100Events_pp2hh4b_14TeV_AUTO.cmd . #some line to auto-generate script

echo "----> Install collier ninja"
./bin/mg5_aMC installPackages1.cmd
echo "----> Install Pythia8 Delphes ExRootAnalysis"
./bin/mg5_aMC installPackages2.cmd
echo "----> Copy over Delphes .tcl files"
cp Delphes/cards/CMS_PhaseII/muonMomentumResolution.tcl Template/Common/Cards/
cp Delphes/cards/CMS_PhaseII/trackMomentumResolution.tcl Template/Common/Cards/
cp Delphes/cards/CMS_PhaseII/muonTightId.tcl Template/Common/Cards/
cp Delphes/cards/CMS_PhaseII/muonLooseId.tcl Template/Common/Cards/
echo "----> sed Delphes card for pileup location"
sed -s 's/\/eos\/cms\/store\/group/\/eos\/uscms\/store\/user\/benjtann/g' Delphes/cards/CMS_PhaseII/CMS_PhaseII_0PU_v02.tcl
echo "----> Try generating some events?"
./bin/mg5_aMC test_100Events_pp2hh4b_14TeV_AUTO.cmd
