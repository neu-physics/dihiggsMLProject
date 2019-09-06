#!/bin/bash

echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node

#source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc62-opt/setup.sh # CentOS7
source /cvmfs/sft.cern.ch/lcg/views/LCG_89/x86_64-slc6-gcc62-opt/setup.sh # SLC6

xrdcp -s root://cmseos.fnal.gov//store/user/benjtann/upgrade/madgraph5/MG5_aMC_v2.6.6.tar.gz .
xrdcp -s root://cmseos.fnal.gov//store/user/benjtann/upgrade/delphes/PhaseII/MinBias_100k.pileup .
#xrdcp -s root://cmseos.fnal.gov//store/user/benjtann/upgrade/madgraph5/installPackages1.cmd .
#xrdcp -s root://cmseos.fnal.gov//store/user/benjtann/upgrade/madgraph5/installPackages2.cmd .
#xrdcp -s root://cmseos.fnal.gov//store/user/benjtann/upgrade/madgraph5/installPackages3.cmd .
#xrdcp -s root://cmseos.fnal.gov//store/user/benjtann/upgrade/madgraph5/finishPythiaInstall.sh .

xrdcp -s root://cmseos.fnal.gov//store/user/benjtann/upgrade/samples/$1/$2 .

#cp ../MG5_aMC_v2.6.6.tar.gz .
tar -xf MG5_aMC_v2.6.6.tar.gz
rm MG5_aMC_v2.6.6.tar.gz
cd MG5_aMC_v2_6_6

cp ../installPackages*.cmd . #some line to wget install script or just pass via condor submission
cp ../pythia8_install.sh . #some line to get kludge-script to finish pythia8 install
#cp ../$2 . #some line to auto-generate script

echo "----> Install collier ninja"
./bin/mg5_aMC installPackages1.cmd
echo "----> Install lhapdf5 zlib hepmc"
./bin/mg5_aMC installPackages2.cmd
echo "----> Install Pythia8 (manual)"
source pythia8_install.sh
echo "----> Install mg5amc_py8_interface Delphes ExRootAnalysis"
./bin/mg5_aMC installPackages3.cmd
echo "----> Copy over Delphes .tcl files"
cp Delphes/cards/CMS_PhaseII/muonMomentumResolution.tcl Template/Common/Cards/
cp Delphes/cards/CMS_PhaseII/trackMomentumResolution.tcl Template/Common/Cards/
cp Delphes/cards/CMS_PhaseII/muonTightId.tcl Template/Common/Cards/
cp Delphes/cards/CMS_PhaseII/muonLooseId.tcl Template/Common/Cards/
echo "----> sed Delphes card for pileup location"
sed -i 's/\/eos\/cms\/store\/group\/upgrade\/delphes\/PhaseII/../g' Delphes/cards/CMS_PhaseII/CMS_PhaseII_0PU_v02.tcl


echo "----> Try generating some events?"
echo $PWD
./bin/mg5_aMC $2


#### Now that the run is over, there is one or more root files created
echo "List all root files = "
ls ./*/Events/run_01*/*.root
echo "List all run files"
ls ./*/Events/run_01*/*
echo "*******************************************"
OUTDIR=root://cmseos.fnal.gov//store/user/benjtann/upgrade/samples/$1/
echo "xrdcp output for condor"
#cat input/mg5_configuration.txt

for FILE in ./input/*.txt
do
  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"
  xrdcp -f ${FILE} ${OUTDIR}/${FILE} 2>&1
  XRDEXIT=$?
  if [[ $XRDEXIT -ne 0 ]]; then
    rm *.root
    echo "exit code $XRDEXIT, failure in xrdcp"
    exit $XRDEXIT
  fi
  rm ${FILE}
done

for FILE in ./*/Events/run_01*/*.root
do
  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"
  xrdcp -f ${FILE} ${OUTDIR}/${FILE} 2>&1
  XRDEXIT=$?
  if [[ $XRDEXIT -ne 0 ]]; then
    rm *.root
    echo "exit code $XRDEXIT, failure in xrdcp"
    exit $XRDEXIT
  fi
  rm ${FILE}
done


cd ${_CONDOR_SCRATCH_DIR}
rm -rf MG5_aMC_v2_6_6

