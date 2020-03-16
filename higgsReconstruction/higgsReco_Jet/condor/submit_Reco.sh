#!/bin/bash
#subfilename=0
mkdir Jet_${1}
mkdir Jet_${1}/out
mkdir Jet_${1}/err
mkdir Jet_${1}/log
mkdir Jet_${1}/sub

if [ ! -d "/eos/user/l/lian/diHiggs/Jet_${1}" ]; then
  mkdir /eos/user/l/lian/diHiggs/Jet_${1}
fi

pkgname="pkg_${1}.tar.gz"
tar -cvzf ${pkgname} ../../../higgsReconstruction ../../../utils  --exclude '*.ipynb' --exclude '*.png' --exclude '*.gif' --exclude 'condor' --exclude '*.csv' --exclude '*.pkl'

xrdcp ${pkgname} root://eosuser.cern.ch//eos/user/l/lian/diHiggs/Jet_${1}/.

cp /tmp/x509up_u119462 /afs/cern.ch/user/l/lian/private/

sub_filename="submit_${1}.jdl"
Proxy_path="/afs/cern.ch/user/l/lian/private/x509up_u119462"
touch $sub_filename
echo universe                = vanilla > $sub_filename
echo executable              = runReco.sh >> $sub_filename
echo arguments               = ${1} ${pkgname} ${Proxy_path} >> $sub_filename
#echo use_x509userproxy 	 = true >> $sub_filename
#echo should_transfer_files 	 = YES >> $sub_filename
#echo transfer_input_files 	 = ${config} >> $sub_filename
echo output                  = Jet_${1}/out/job.out >> $sub_filename
echo error                   = Jet_${1}/err/job.err >> $sub_filename
echo log                     = Jet_${1}/log/job.log >> $sub_filename
#echo request_cpus            = 8 >> $sub_filename
echo request_cpus            = 2 >> $sub_filename
echo +MaxRuntime 		 = 86400 >> $sub_filename
#echo 'requirements            = (OpSysAndVer =?= "SLCern6")' >> $sub_filename
#echo '+JobFlavour             = "tomorrow"' >> $sub_filename
echo queue >> $sub_filename

condor_submit $sub_filename
    
mv *jdl Jet_${1}/sub
mv ${pkgname} Jet_${1}/sub
