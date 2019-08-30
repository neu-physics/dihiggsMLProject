source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc62-opt/setup.csh
cp ../MG5_aMC_v2.6.6.tar.gz .
tar -xvzf MG5_aMC_v2.6.6.tar.gz
cd MG5_aMC_v2_6_6

cp ../installPackages*.cmd . #some line to wget install script or just pass via condor submission

echo "----> Install collier ninja"
./bin/mg5_aMC installPackages1.cmd
echo "----> Install Pythia8 Delphes ExRootAnalysis"
./bin/mg5_aMC installPackages2.cmd
echo "----> Try some new stuff"
