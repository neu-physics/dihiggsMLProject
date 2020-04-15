# /usr/bin/python

#Author: Ben Tannenwald
#Date: April 14th, 2020
#Purpose: Script to submit condor jobs for all files in sample

import os, sys, argparse
import numpy as np

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--outputDir", help="output directory for processed histograms and roofiles")
parser.add_argument("--inputTXTFile", help=".txt file containing input pkl files")
parser.add_argument("--nOutputFiles", help="number of desired output files")
args = parser.parse_args()

if ( len(vars(args)) != 3 ): # 2 OR 3 --> three for options
    os.system('python submitSampleToCondor.py -h')
    quit()

# ** A. Test output directory existence and create if DNE
if(args.outputDir is None):
    print( "#### Need to specify output directory using --outputDir <desired output directory> ####\nEXITING")
    quit()
else:
    if ( not os.path.exists(args.outputDir) ):
        print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(args.outputDir))
        os.system("mkdir {0}".format(args.outputDir))

    if ( not os.path.exists( (args.outputDir + '/condor_logs/') ) ):
        os.system("mkdir {0}".format( (args.outputDir + '/condor_logs/')) )
    if ( not os.path.exists( (args.outputDir + '/condor_err/') ) ):
        os.system("mkdir {0}".format( (args.outputDir + '/condor_err/')) )
    if ( not os.path.exists( (args.outputDir + '/condor_out/') ) ):
        os.system("mkdir {0}".format( (args.outputDir + '/condor_out/')) )

    print( '-- Setting outputDir = {0}'.format(args.outputDir))

# ** B. Test input .txt file and exit if DNE
if(args.inputTXTFile is None):
    print( "#### Need to specify input .cmd file using --inputTXTFile <address to .cmd file> ####\nEXITING\n")
    quit()
else:
    if ( not os.path.exists(args.inputTXTFile) ):
        print( "#### Specified input file ({0}) DNE ####.\nEXITING\n".format(args.inputTXTFile))
        quit()
    else:
        print( '-- Setting inputTXTFile = {0}'.format(args.inputTXTFile))

# ** C. Test nOutputFiles and exit if not sensible
if (args.nOutputFiles).isdigit() is True:
    print( "-- Running with {0} outputfiles\n".format(args.nOutputFiles) )
else:
    print( "#### WARNING: Passed option of {0} number of output files makes no sense. DNE ####\nEXITING\n".format(args.nOutputFiles))
    quit()

# FIXME, 05-08-19 BBT
## ** D. Exit if no grid proxy
#if ( not os.path.exists(os.path.expandvars("$X509_USER_PROXY")) ):
#    print "#### No GRID PROXY detected. Please do voms-proxy-init -voms cms before submitting Condor jobs ####.\nEXITING"
#    quit()


# *** 1. Create output directory in personal EOS
if ( not os.path.exists("/eos/uscms/store/user/benjtann/upgrade/samples/{0}/".format(args.outputDir)) ):
    os.system("mkdir /eos/uscms/store/user/benjtann/upgrade/samples/{0}/".format(args.outputDir))

# *** 2. Create temporary .jdl file for condor submission
print( "\n##########     Submitting Condor jobs     ##########\n")

# *** 3. Figure out how many files per job
with open(args.inputTXTFile, 'r') as f:
    pklFileList = f.readlines()

stepSize = np.ceil( len(pklFileList)/float(args.nOutputFiles) )

# *** 4. Loop over file chunks
for startFile in np.arange(0, len(pklFileList), stepSize):

    # ** A. Calculate upper limit of file range
    endFile = min( (startFile + stepSize ), len(pklFileList)  )
    iteration = str(int(startFile / stepSize))

    # ** B. Make temporary subset .txt
    tempFileList = pklFileList[int(startFile):int(endFile)]
    tempTxtName = "{}/temp_{}.txt".format(args.outputDir, iteration)
    os.system( "touch {}".format( tempTxtName ) )
    with open( '{}'.format(tempTxtName), 'w') as f:
        for t_file in tempFileList:
            f.write( t_file )

    # ** B. Submit a job
    jdl_filename = "submitOneImageMaker_{0}_{1}.jdl".format(args.outputDir, iteration)

    os.system("touch {0}".format(jdl_filename))
    os.system("echo universe = vanilla > {0}".format(jdl_filename))
    os.system("echo Executable = imageProducer.csh >> {0}".format(jdl_filename))
    os.system("echo Should_Transfer_Files = YES >> {0}".format(jdl_filename))
    os.system("echo WhenToTransferOutput = ON_EXIT >> {0}".format(jdl_filename))
    os.system("echo request_cpus = 2 >> {0}".format(jdl_filename))
    os.system("echo Transfer_Input_Files = imageProducer.csh, ../../higgsReconstruction/JetCons.py, ../imageMaker.py, {} >> {}".format(tempTxtName, jdl_filename))
    os.system("echo Output = {0}/condor_out/outfile_{1}.out  >> {2}".format( args.outputDir, iteration, jdl_filename))
    os.system("echo Error = {0}/condor_err/outfile_{1}.err >> {2}".format(args.outputDir , iteration, jdl_filename))
    os.system("echo Log = {0}/condor_logs/outfile_{1}.log >> {2}".format(args.outputDir , iteration, jdl_filename))
    os.system("echo x509userproxy = ${{X509_USER_PROXY}} >> {0}".format(jdl_filename))
    os.system("echo Arguments = {0} {1} {2} >> {3}".format( tempTxtName, str(args.outputDir + '_' + iteration), args.outputDir, jdl_filename)) 
    os.system("echo Queue 1 >> {0}".format(jdl_filename))       
    #os.system("""echo +DesiredOS="SL7" >> {}""".format(jdl_filename))
    os.system("condor_submit {0}".format(jdl_filename))

# *** 3. Cleanup submission directory
print( "\n##########     Cleanup submission directory     ##########\n")
os.system("rm *.jdl") # remove temp condor submission scripts

