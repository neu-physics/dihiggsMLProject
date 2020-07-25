#!/bin/bash

echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node

#source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh # idk, some LPC stuff
source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh # CentOS 7, Python 3
#source /cvmfs/sft.cern.ch/lcg/views/LCG_97/x86_64-centos7-gcc8-opt/setup.csh # CentOS 7, Python 2
#source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc62-opt/setup.sh # CentOS7

echo "----> List all python headers locally"
ls -a 

echo "----> Copy input file locally"
xrdcp -s root://cmseos.fnal.gov/$1 .
touch temp.txt
echo $PWD/*.root > temp.txt
echo "----> List input file locally"
ls -a  *.root

echo "----> make local python venv"
ENVNAME="recoenv"
python -m venv ${ENVNAME}
source ${ENVNAME}/bin/activate
python -m pip install uproot
python -m pip install pandas
#pip install pyyaml
#pip install sklearn
#pip install xgboost
#pip install pyswarms
#pip install pandas
#pip --user install uproot

echo "----> Try reconstructing some events?"
echo $PWD
python runEventReconstruction.py --inputTXTFile temp.txt --outputTag $2


#### Now that the run is over, there is one or more root files created
echo "List all the things = "
ls ./*
ls ./$2/*

echo "List all .csv files = "
ls ./$2/*.csv
echo "List all .pkl files"
ls ./$2/*.pkl
echo "*******************************************"
OUTDIR=root://cmseos.fnal.gov//store/user/benjtann/upgrade/samples/$3/

echo ">> xrdcp csv files"
for FILE in ./$2/*.csv
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


echo " >> xrdcp pkl files"
for FILE in ./$2/*.pkl
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

echo " >> cleanup and close"
deactivate
cd ${_CONDOR_SCRATCH_DIR}
rm -rf ${ENVNAME}

