# /usr/bin/python

#Author: Ben Tannenwald
#Date: April 13th, 2020
#Purpose: Script to submit condor jobs for all files in sample

import os,sys, argparse

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--outputDir", help="output directory for processed histograms and roofiles")
parser.add_argument("--inputTXTFile", help=".txt file containing input delphes files")
args = parser.parse_args()

if ( len(vars(args)) != 2 ): # 2 OR 3 --> three for options
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

nFilesSubmitted = 0

with open(args.inputTXTFile, 'r') as f:
    delphesFileList = f.readlines()

for delphesFile in delphesFileList:
    delphesFile = delphesFile.split('\n')[0]
    eventRange = delphesFile.split('/')[7]
    #print(delphesFile, eventRange)

    # ** A. Make subdirectory for eventRange if doesn't already exist
    if ( not os.path.exists("/eos/uscms/store/user/benjtann/upgrade/samples/{0}".format(args.outputDir)) ):
        os.system("mkdir /eos/uscms/store/user/benjtann/upgrade/samples/{0}/".format(args.outputDir) )


    # ** B. Submit a job
    jdl_filename = "submitOneReconstruction_{0}_{1}.jdl".format(args.outputDir, eventRange)

    os.system("touch {0}".format(jdl_filename))
    os.system("echo universe = vanilla > {0}".format(jdl_filename))
    os.system("echo Executable = reconstructOneFile.csh >> {0}".format(jdl_filename))
    os.system("echo Should_Transfer_Files = YES >> {0}".format(jdl_filename))
    os.system("echo WhenToTransferOutput = ON_EXIT >> {0}".format(jdl_filename))
    os.system("echo request_cpus = 2 >> {0}".format(jdl_filename))
    os.system("echo Transfer_Input_Files = reconstructOneFile.csh, ../runEventReconstruction.py, ../eventReconstructionClass.py, ../JetCons.py >> {0}".format(jdl_filename))
    os.system("echo Output = {0}/condor_out/outfile_{1}.out  >> {2}".format(args.outputDir, eventRange, jdl_filename))
    os.system("echo Error = {0}/condor_err/outfile_{1}.err >> {2}".format(args.outputDir, eventRange, jdl_filename))
    os.system("echo Log = {0}/condor_logs/outfile_{1}.log >> {2}".format(args.outputDir, eventRange, jdl_filename))
    os.system("echo x509userproxy = ${{X509_USER_PROXY}} >> {0}".format(jdl_filename))
    #print("echo Arguments = {0} {1} {2} >> {3}".format( delphesFile, str(args.outputDir + '_' + eventRange), args.outputDir, jdl_filename))
    os.system("echo Arguments = {0} {1} {2} >> {3}".format( delphesFile, str(args.outputDir + '_' + eventRange), args.outputDir, jdl_filename))
    os.system("echo Queue 1 >> {0}".format(jdl_filename))       
    #os.system("""echo +DesiredOS="SL7" >> {}""".format(jdl_filename))
    os.system("condor_submit {0}".format(jdl_filename))


# *** 3. Cleanup submission directory
print( "\n##########     Cleanup submission directory     ##########\n")
os.system("rm *.jdl")

