#!/bin/bash

echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node

echo "---> default python"
pyVer=$(python --version)

#### Script for transfering files locally ####
#### FRAMEWORK SANDBOX SETUP ####
echo "---> Source environment for CMSSW"

# Load cmssw_setup function
source cmssw_setup.sh

# Setup CMSSW Base
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
source $VO_CMS_SW_DIR/cmsset_default.sh

# Download sandbox
sandbox_name="sandbox-CMSSW_10_2_22-d5a51a1.tar.bz2"
wget --no-check-certificate --progress=bar "http://stash.osgconnect.net/+benjtann/${sandbox_name}" || exit_on_error $? 150 "Could not download sandbox."

# Setup framework from sandbox
cmssw_setup $sandbox_name || exit_on_error $? 151 "Could not unpack sandbox"

echo "---> CMSSW python"
pyVer=$(python --version)

#### END OF FRAMEWORK SANDBOX SETUP ####

#echo "----> List all python headers locally"
#ls -a 

echo "----> Copy input signal file(s) locally"
mkdir signal
while IFS="" read -r p || [ -n "$p" ]
do
  #filepath='root://stash.osgconnect.net:1094/'
  filepath='root://cmseos.fnal.gov:1094/'
  filepath+=$p
  echo $filepath
  xrdcp -s $filepath .
done < ./$2
mv *.h5 signal/

touch tempSignalFiles.txt
ls $PWD/signal/*.h5 > tempSignalFiles.txt
echo "----> List input signal file(s) locally"
cat tempSignalFiles.txt

echo "----> Copy input background file(s) locally"
mkdir background
while IFS="" read -r p || [ -n "$p" ]
do
  filepath='root://cmseos.fnal.gov:1094/'
  filepath+=$p
  echo $filepath
  xrdcp -s $filepath .
done < ./$3
mv *.h5 background/

touch tempBackgroundFiles.txt
ls $PWD/background/*.h5 > tempBackgroundFiles.txt
echo "----> List input background file(s) locally"
cat tempBackgroundFiles.txt

echo "---> End CMSSW"

#### GPU SETUP ####
echo "----> Setup GPU packages"
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh # idk, some LPC stuff
echo python --version
pyVer=$(python --version)

mlenv="mlenv1"
#eval "bash activate $mlenv"
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/anaconda3/bin/activate $mlenv

echo "---> TF python"
pyVer=$(python --version)

echo "----> make utils subdir"
mkdir utils
mv commonFunctions.py utils/
ls $PWD/utils

echo "----> run cnn wrapper"
echo $PWD
pythonCMD="python cnnMultiWrapper.py --outputDir ${1} --inputHHFile tempSignalFiles.txt  --inputQCDFile tempBackgroundFiles.txt --imageCollections ${5}"

extras=$(head -n 1 $4)
if [ "$extras" != "None" ]; then 
    pythonCMD+=" --extraVariables ${extras}"
fi
if [ "$6" == "True" ]; then 
    pythonCMD+=" --addClassWeights"
fi
if [ "$7" == "True" ]; then 
    pythonCMD+=" --testRun"
fi
if [ "$8" == "True" ]; then 
    pythonCMD+=" --condorRun"
fi

echo $pythonCMD


# Execute in background
#python myScript.py <script options> &
eval $pythonCMD #&
## Store pid
#mypid=$!
# Wait for process to finish
#wait $pid
#exitcode=$?
#
#if [ $exitcode != 0 ]; then
#    echo "Python script error code: $exitcode"
#    # Try to print information from dmesg"
#    dmesg | grep $pid 2>&1
#fi

echo "----> Post imaging"


#### Now that the run is over, there is one or more root files created
echo "List all the things = "
ls ./*
ls ./$1/*
ls ./$1/*/*
ls ./$1/*/*/*

echo "List all .h5 files = "
ls ./$1/models/*.hdf5
echo "*******************************************"
OUTDIR=root://cmseos.fnal.gov//store/user/benjtann/upgrade/cnn/

xrdcp -r -p $1 $OUTDIR/

echo " >> cleanup and close"
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/anaconda3/bin/deactivate

#cd ${_CONDOR_SCRATCH_DIR}
rm signal -rf
rm background -rf
rm utils -rf
#rm *.py
rm temp*.txt
rm sandbox*
