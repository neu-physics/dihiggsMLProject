#!/bin/bash

echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node

echo "---> default python"
pyVer=$(python --version)

#### Script for transfering files locally ####
#### FRAMEWORK SANDBOX SETUP ####
echo "---> Source environment for CMSSW"
sh transferFiles.sh

echo "---> End CMSSW"

#### GPU SETUP ####
echo "----> Setup GPU packages"
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh # idk, some LPC stuff
pyVer=$(python --version)

source activate mlenv1
#source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh # CentOS 7, Python 3
#source /cvmfs/sft.cern.ch/lcg/views/LCG_97/x86_64-centos7-gcc8-opt/setup.csh # CentOS 7, Python 2
#source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc62-opt/setup.sh # CentOS7

echo "---> TF python"
pyVer=$(python --version)

echo "----> run cnn wrapper"
echo $PWD
pythonCMD="python cnnMultiWrapper.py --outputDir ${1} --inputHHFile tempSignalFiles.txt  --inputQCDFile tempBackgroundFiles.txt --imageCollections ${5}"
if [ "$4" != "[]" ]; then 
    pythonCMD+=" --extraVariables ${4}"
fi
if [ $6 == "True" ]; then 
    pythonCMD+=" --addClassWeights"
fi
if [ $7 == "True" ]; then 
    pythonCMD+=" --testRun"
fi
if [ $8 == "True" ]; then 
    pythonCMD+=" --condorRun"
fi

echo $pythonCMD
eval $pythonCMD

echo "----> Post imaging"


#### Now that the run is over, there is one or more root files created
echo "List all the things = "
ls ./*
ls ./$1/*
ls ./$1/*/*
ls ./$1/*/*/*

echo "List all .h5 files = "
ls ./$2/images/*.h5
echo "*******************************************"
OUTDIR=root://cmseos.fnal.gov//store/user/benjtann/upgrade/cnn/

xrdcp -r -p $1 $OUTDIR/

echo " >> cleanup and close"
source deactivate
#cd ${_CONDOR_SCRATCH_DIR}
rm signal -rf
rm background -rf
#rm *.py
#rm *.txt

