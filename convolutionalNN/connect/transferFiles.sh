#### FRAMEWORK SANDBOX SETUP ####
echo "---> Enter bash for CMSSW"

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

# Enter script directory
#cd $CMSSW_BASE/src/

echo "---> CMSSW python"
echo python --version
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

echo "---> Exit bash for CMSSW"
