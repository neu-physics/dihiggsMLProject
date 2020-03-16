#!/bin/bash                                                                  

export X509_USER_PROXY=${3}
voms-proxy-info -all
voms-proxy-info -all -file ${3}

echo "Starting job on " `date` #Date/time of start of job                    
echo "Running on: `uname -a`" #Condor job is running on this node            
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
source /cvmfs/cms.cern.ch/cmsset_default.sh  ## if a tcsh script, use .csh instead of .sh

xrdcp root://eosuser.cern.ch//eos/user/l/lian/diHiggs/Jet_${1}/${2} .
tar -xf ${2}
echo "ls -l $PWD"
ls -l $PWD
rm ${2}

virtualenv -p `which python3` venv
source venv/bin/activate
pip install pyyaml
pip install sklearn
pip install xgboost
pip install pyswarms
pip install pandas
pip install uproot
#pip install ffmpeg

sleep 10

cd higgsReconstruction/higgsReco_Jet/
python3 runRecoForManyFiles.py --input pp2hh4b_file_eos.txt --diHiggs True --inputTag pp2hh4b_1M

echo "ls -l $PWD"
ls -l $PWD

xrdcp -r *pp2hh4b_1M* root://eosuser.cern.ch//eos/user/l/lian/diHiggs/Jet_${1}

sleep 10
deactivate
