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

echo "----> Copy input file(s) locally"
while IFS="" read -r p || [ -n "$p" ]
do
  filepath='root://cmseos.fnal.gov/'
  filepath+=$p
  echo $filepath
  xrdcp -s $filepath .
done < ./$1

touch tempCondorFiles.txt
ls $PWD/*.pkl > tempCondorFiles.txt
echo "----> List input file(s) locally"
ls -a  *.pkl

echo "----> make local python venv"
ENVNAME="imageEnv"
python -m venv ${ENVNAME}
source ${ENVNAME}/bin/activate
python -m pip install uproot

echo "----> create temp wrapper"
touch temp_wrapper.py
echo "#!/usr/bin/python" >> temp_wrapper.py
echo "from imageMaker import imageMaker" >> temp_wrapper.py
echo "imager = imageMaker('$2', 'tempCondorFiles.txt')" >> temp_wrapper.py
echo "imager.processFiles()" >> temp_wrapper.py

echo "----> Try imaging some events"
echo $PWD
python temp_wrapper.py 

echo "----> Post imaging"


#### Now that the run is over, there is one or more root files created
echo "List all the things = "
ls ./*
ls ./$2/*

echo "List all .h5 files = "
ls ./$2/images/*.h5
echo "*******************************************"
OUTDIR=root://cmseos.fnal.gov//store/user/benjtann/upgrade/samples/$3/

echo ">> xrdcp h5 files"
for FILE in ./$2/images/*.h5
do
  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"
  xrdcp -f ${FILE} ${OUTDIR}/${FILE} 2>&1
  XRDEXIT=$?
  if [[ $XRDEXIT -ne 0 ]]; then
    rm *.h5
    echo "exit code $XRDEXIT, failure in xrdcp"
    exit $XRDEXIT
  fi
  rm ${FILE}
done

echo ">> xrdcp .png plots"
ls ./$2/plots/*
for FILE in ./$2/plots/*.png
do
  echo "xrdcp -f ${FILE} ${OUTDIR}/${FILE}"
  xrdcp -f ${FILE} ${OUTDIR}/${FILE} 2>&1
  XRDEXIT=$?
  if [[ $XRDEXIT -ne 0 ]]; then
    rm *.png
    echo "exit code $XRDEXIT, failure in xrdcp"
    exit $XRDEXIT
  fi
  rm ${FILE}
done

echo " >> cleanup and close"
deactivate
cd ${_CONDOR_SCRATCH_DIR}
rm tempCondorFiles.txt
rm temp_wrapper.py
rm -rf ${ENVNAME}
rm -rf *.pkl

