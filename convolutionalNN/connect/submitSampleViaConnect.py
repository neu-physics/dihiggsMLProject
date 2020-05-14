# /usr/bin/python
#Author: Ben Tannenwald
#Date: May 8th, 2020
#Purpose: Script to submit condor jobs via CMS CONNECT

import os, sys, argparse


# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--outputDir", help="output directory for model training outputs")
parser.add_argument("--inputHHFile", help=".txt file containing input .h5 files for dihiggs")
parser.add_argument("--inputQCDFile", help=".txt file containing input .h5 files for qcd")
parser.add_argument('-x','--extraVariables', nargs='+', help='add extra variables to concatenate with convolutional outputs')
parser.add_argument('-ic','--imageCollections', nargs='+', help='imageCollections to process', required=True)
parser.add_argument('--addClassWeights', dest='addClassWeights', action='store_true')
parser.add_argument('--testRun', dest='testRun', action='store_true')
parser.add_argument('--condorRun', dest='condorRun', action='store_true')
parser.set_defaults(addClassWeights=False)
parser.set_defaults(testRun=False)
parser.set_defaults(condorRun=False)

args = parser.parse_args()


if ( len(vars(args)) != 8 ): # --> depends on default options
    os.system('python submitSampleViaConnect.py -h')
    print( vars(args), len(vars(args)))
    quit()

# ** A. Test output directory existence and create if DNE
if(args.outputDir is None):
    print( "#### Need to specify output directory using --outputDir <desired output directory> ####\nEXITING")
    quit()
else:
    if ( not os.path.exists(args.outputDir) ):
        print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(args.outputDir))
        os.system("mkdir {0}".format(args.outputDir))
        os.system("mkdir {0}/logs".format(args.outputDir))

    print( '-- Setting outputDir = {0}'.format(args.outputDir))

# ** B. Test input .txt file and exit if DNE
if(args.inputHHFile is None):
    print( "#### Need to specify input .txt file using --inputHHFile <address to .txt file> ####\nEXITING\n")
    quit()
else:
    if ( not os.path.exists(args.inputHHFile) ):
        print( "#### Specified input file ({0}) DNE ####.\nEXITING\n".format(args.inputHHFile))
        quit()
    else:
        print( '-- Setting inputHHFile = {0}'.format(args.inputHHFile))

# ** C. Test input .txt file and exit if DNE
if(args.inputQCDFile is None):
    print( "#### Need to specify input .txt file using --inputQCDFile <address to .txt file> ####\nEXITING\n")
    quit()
else:
    if ( not os.path.exists(args.inputQCDFile) ):
        print( "#### Specified input file ({0}) DNE ####.\nEXITING\n".format(args.inputQCDFile))
        quit()
    else:
        print( '-- Setting inputQCDFile = {0}'.format(args.inputQCDFile))


# ** D. Test image collections and quit if empty
if(args.imageCollections is None):
    print( "#### Need to specify at least one image collection using --imageCollections  <collections separated by spaces> ####\nEXITING")
    quit()
else:
    print( '-- Setting imageCollections = {0}'.format(args.imageCollections))

# multi-run
#imageCollections = ['compositeImgs',
#                    'compositeImgs_lessThan4j', 'compositeImgs_ge4jInclb',
                    #'compositeImgs_ge4j0b', 'compositeImgs_ge4j1b','compositeImgs_ge4j2b','compositeImgs_ge4j3b','compositeImgs_ge4j4b','compositeImgs_ge4jge4b', 
                    #'compositeImgs_HT150','compositeImgs_HT300', 'compositeImgs_HT450',
                    #'trackImgs', 'nHadronImgs', 'photonImgs',
#]


# ** E. Test output directory existence and create if DNE
if(args.extraVariables is None):
    print( '-- Setting extraVariables = {0}'.format(args.extraVariables))
    args.extraVariables = []
else:
    print( '-- Setting extraVariables = {0}'.format(args.extraVariables))


# *** 1. Reformat passed lists to fit with BASH
_extraVariables = ["'{0}'".format(var) for var in args.extraVariables]
_extraVariables = " ".join( _extraVariables ) 


# *** 2. Loop over collections to submit
for iCollection in range(0, len(args.imageCollections)):
    imageCollection = args.imageCollections[iCollection]

    # ** A. Create temp bash script to process job
    tempBashScript = "trainModel.sh" # this should actually be globally flex

    # ** B. Create condor submission .jdl
    jdl_filename = "processOneModel_{0}_{1}.jdl".format(args.outputDir, imageCollection)

    os.system("touch {0}".format(jdl_filename))
    os.system("echo universe = vanilla > {0}".format(jdl_filename))
    os.system("echo should_transfer_files = YES >> {0}".format(jdl_filename))
    os.system("echo transfer_input_files = /etc/ciconnect/templates/cmssw_setup.sh, transferFiles.sh, ../cnnModelClass.py, ../../utils/commonFunctions.py, ../cnnMultiWrapper.py, {0}, {1}, {2}, >> {3}".format(tempBashScript, args.inputHHFile, args.inputQCDFile, jdl_filename))
    os.system("echo Executable = {0} >> {1}".format(tempBashScript, jdl_filename))
    os.system("echo Output = {0}/logs/job_{1}.out  >> {2}".format( args.outputDir, imageCollection, jdl_filename))
    os.system("echo Error = {0}/logs/job_{1}.err >> {2}".format(args.outputDir, imageCollection, jdl_filename))
    os.system("echo Log = {0}/logs/job_{1}.log >> {2}".format(args.outputDir, imageCollection, jdl_filename))
    
    os.system("""echo Arguments = {0} {1} {2} "{3}" {4} {5} {6} {7} >> {8}""".format( args.outputDir, args.inputHHFile.split('/')[-1], args.inputQCDFile.split('/')[-1], _extraVariables, imageCollection, args.addClassWeights, args.testRun, args.condorRun, jdl_filename)) 

    os.system("""echo +DesiredOS="SL7" >> {0}""".format(jdl_filename))
    #os.system("""echo +ProjectName="cms-org-cern" >> {}""".format(jdl_filename))
    #os.system("echo x509userproxy = ${{X509_USER_PROXY}} >> {0}".format(jdl_filename))

    # Request GPUs for CNN jobs
    #os.system("""echo Requirements = HAS_SINGULARITY == True && CUDACapability >= 3 >> {0}""".format(jdl_filename))
    os.system("""echo "requirements = CUDACapability >= 3" >> {0}""".format(jdl_filename))
    os.system("echo request_memory = 4 Gb >> {0}".format(jdl_filename))
    os.system("echo request_cpus = 1 >> {0}".format(jdl_filename))
    os.system("echo request_gpus = 1 >> {0}".format(jdl_filename))
    os.system('''echo +SingularityImage = \\"/cvmfs/singularity.opensciencegrid.org/opensciencegrid/tensorflow-gpu:latest\\" >> {0}'''.format(jdl_filename))

    # Request CPUs for normal jobs
    #os.system("echo request_cpus = 2 >> {0}".format(jdl_filename))
    
    os.system("echo Queue 1 >> {0}".format(jdl_filename))       
    os.system("condor_submit {0}".format(jdl_filename))



# *** 3. Cleanup submission directory
print( "\n##########     Cleanup submission directory     ##########\n")
os.system("rm *.jdl") # remove temp condor submission scripts
