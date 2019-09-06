# /usr/bin/python

#Author: Ben Tannenwald
#Date: August 5th, 2019
#Purpose: Script to submit condor jobs for all files in sample

import os,sys, argparse

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--outputDir", help="output directory for processed histograms and roofiles")
parser.add_argument("--inputCMDFile", help=".cmd file containing generation command")
parser.add_argument("--nTotalEvents", help="total number of events to generate", default='5000')
parser.add_argument("--nEventsPerJob", help="number of events to run per job... use if you want to split a large generation into several sub-runs", default='1000')
args = parser.parse_args()

if ( len(vars(args)) != 4 ): # 2 OR 3 --> three for options
    os.system('python submitSampleToCondor.py -h')
    quit()

# ** A. Test output directory existence and create if DNE
if(args.outputDir is None):
    print "#### Need to specify output directory using --outputDir <desired output directory> ####\nEXITING"
    quit()
else:
    if ( not os.path.exists(args.outputDir) ):
        print "Specified output directory ({0}) DNE.\nCREATING NOW".format(args.outputDir)
        os.system("mkdir {0}".format(args.outputDir))

    if ( not os.path.exists( (args.outputDir + '/condor_logs/') ) ):
        os.system("mkdir {0}".format( (args.outputDir + '/condor_logs/')) )
    if ( not os.path.exists( (args.outputDir + '/condor_err/') ) ):
        os.system("mkdir {0}".format( (args.outputDir + '/condor_err/')) )
    if ( not os.path.exists( (args.outputDir + '/condor_out/') ) ):
        os.system("mkdir {0}".format( (args.outputDir + '/condor_out/')) )

    print '-- Setting outputDir = {0}'.format(args.outputDir)

# ** B. Test input .txt file and exit if DNE
if(args.inputCMDFile is None):
    print "#### Need to specify input .cmd file using --inputCMDFile <address to .cmd file> ####\nEXITING\n"
    quit()
else:
    if ( not os.path.exists(args.inputCMDFile) ):
        print "#### Specified input file ({0}) DNE ####.\nEXITING\n".format(args.inputCMDFile)
        quit()
    else:
        print '-- Setting inputCMDFile = {0}'.format(args.inputCMDFile)

# ** C. Test nEventsPerJob flag and exit if not sensible
if (args.nEventsPerJob).isdigit() is True:
    print "-- Running with {0} events per job\n".format(args.nEventsPerJob)
else:
    print "#### WARNING: Passed option of {0} files per job makes no sense. DNE ####\nEXITING\n".format(args.nEventsPerJob)
    quit()

# ** C. Test nTotalEvents flag and exit if not sensible
if (args.nTotalEvents).isdigit() is True:
    print "-- Running with {0} events total\n".format(args.nTotalEvents)
else:
    print "#### WARNING: Passed option of {0} total events makes no sense. DNE ####\nEXITING\n".format(args.nTotalEvents)
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
print "\n##########     Submitting Condor jobs     ##########\n"

nEventsSubmitted = 0
nEventsPerJob = int(args.nEventsPerJob)

while( nEventsSubmitted < int(args.nTotalEvents)):

    # ** A. Make CMD file for subjob and copy to correct EOS folder
    subjobRange = '{0}-{1}'.format(nEventsSubmitted, (nEventsSubmitted + nEventsPerJob) )
    subjobCMDFile = '{0}_{1}.cmd'.format( str(args.inputCMDFile)[:-4], subjobRange) 
    os.system("cp {0} {1}".format( args.inputCMDFile, subjobCMDFile ))
    os.system("sed -i 's/set nevents /set nevents {0}/g' {1}".format( nEventsPerJob, subjobCMDFile ))
    os.system("sed -i 's/set iseed /set iseed {0}/g' {1}".format( int(nEventsSubmitted / nEventsPerJob) , subjobCMDFile ))

    if ( not os.path.exists("/eos/uscms/store/user/benjtann/upgrade/samples/{0}/{1}".format(args.outputDir, subjobRange)) ):
        os.system("mkdir /eos/uscms/store/user/benjtann/upgrade/samples/{0}/{1}".format(args.outputDir, subjobRange))

    os.system("xrdcp {0} root://cmseos.fnal.gov//store/user/benjtann/upgrade/samples/{1}/{2}/".format( subjobCMDFile, args.outputDir, subjobRange))

    
    # ** B. Submit a job
    jdl_filename = "submitOneMadgraph_{0}_{1}.jdl".format(args.outputDir, subjobRange)

    os.system("touch {0}".format(jdl_filename))
    os.system("echo universe = vanilla > {0}".format(jdl_filename))
    os.system("echo Executable = madgraphGenerate.csh >> {0}".format(jdl_filename))
    os.system("echo Should_Transfer_Files = YES >> {0}".format(jdl_filename))
    os.system("echo WhenToTransferOutput = ON_EXIT >> {0}".format(jdl_filename))
    os.system("echo request_cpus = 2 >> {0}".format(jdl_filename))
    #os.system("echo Transfer_Input_Files = installPackages1.cmd installPackages2.cmd installPackages3.cmd pythia8_install.sh madgraphGenerate.csh >> {0}".format(jdl_filename))
    os.system("echo Transfer_Input_Files = madgraphGenerate.csh >> {0}".format(jdl_filename))
    os.system("echo Output = {0}/condor_out/outfile_{1}.out  >> {2}".format(args.outputDir, subjobRange, jdl_filename))
    os.system("echo Error = {0}/condor_err/outfile_{1}.err >> {2}".format(args.outputDir, subjobRange, jdl_filename))
    os.system("echo Log = {0}/condor_logs/outfile_{1}.log >> {2}".format(args.outputDir, subjobRange, jdl_filename))
    os.system("echo x509userproxy = ${{X509_USER_PROXY}} >> {0}".format(jdl_filename))
    os.system("echo Arguments = {0} {1} >> {2}".format( str(args.outputDir+'/'+subjobRange), subjobCMDFile, jdl_filename))
    os.system("echo Queue 1 >> {0}".format(jdl_filename))       
    os.system("condor_submit {0}".format(jdl_filename))

    # ** C. Iterate number of events submitted
    nEventsSubmitted += nEventsPerJob


# *** 3. Cleanup submission directory
print "\n##########     Cleanup submission directory     ##########\n"
os.system("rm *.jdl")
os.system("rm *_*-*.cmd")
