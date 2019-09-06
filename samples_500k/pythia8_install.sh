#!/bin/bash

INSTALLPATH=$PWD/HEPTools/pythia8/
MG5PATH=$PWD/

echo " >> Download PYTHIA8"
wget http://home.thep.lu.se/~torbjorn/pythia8/pythia82.tgz
echo " >> Unpack PYTHIA8"
tar -xf pythia82.tgz


echo " >> Enter PYTHIA8 directory"
cd pythia8243

echo " >> Configure PYTHIA8"
make distclean
./configure --prefix=$INSTALLPATH --with-hepmc2=$MG5PATH/HEPTools/hepmc --with-hepmc2-include=$MG5PATH/HEPTools/hepmc/include --with-gzip=$MG5PATH/HEPTools/zlib --with-lhapdf5=$MG5PATH/HEPTools/lhapdf5 --with-boost=/cvmfs/sft.cern.ch/lcg/releases/Boost/1.62.0-c1b05/x86_64-slc6-gcc62-opt --cxx-common='-ldl -fPIC -lstdc++ -DHEPMC2HACK'

# Small fix for Pythia8.2 version. This is harmless to subsequent versions
unamestr=`uname`
echo $CXX
if [[ "$unamestr" == 'Darwin' && "$CXX" != 'clang' ]]; then
    sed -i '' 's/-Qunused-arguments//g' Makefile.inc 
fi


echo " >> Compile PYTHIA8"
make


echo " >> Install PYTHIA8"
mkdir -p ../HEPTools/pythia8//bin ../HEPTools/pythia8//include ../HEPTools/pythia8//lib ../HEPTools/pythia8//share/Pythia8
rm -f ../HEPTools/pythia8//lib/libpythia8.so
cp -a bin/* ../HEPTools/pythia8//bin/
rm -rf ../HEPTools/pythia8//bin/.svn
cp -a include/* ../HEPTools/pythia8//include/
rm -rf ../HEPTools/pythia8//include/.svn
cp -a lib/* ../HEPTools/pythia8//lib/
rm -rf ../HEPTools/pythia8//lib/.svn
cp -a share/Pythia8/* ../HEPTools/pythia8//share/Pythia8/
rm -rf ../HEPTools/pythia8//share/Pythia8/.svn
cp -a examples ../HEPTools/pythia8//share/Pythia8/
rm -rf ../HEPTools/pythia8//share/Pythia8/examples/.svn

echo ">> Finished make install, cd to INSTALLPATH"
echo $INSTALLPATH
cd $INSTALLPATH/bin
sed -e s/"if \[ \"\$VAR\" = \"LDFLAGS\" ]\; then OUT+=\" -ld\""/"if \[ \"\$VAR\" = \"LDFLAGS\" ]\; then OUT+=\" -ldl\""/g pythia8-config> pythia8-config.tmp
mv pythia8-config.tmp pythia8-config
chmod ug+x pythia8-config

echo ">> Finished PYTHIA8 installation, moving to MG5PATH"
echo $MG5PATH
cd $MG5PATH