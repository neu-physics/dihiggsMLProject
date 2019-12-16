# /usr/bin/python

#Author: Ben Tannenwald
#Date: December 13th, 2019
#Purpose: Script to run event reconstruction on condor-split .root output files

import os,sys, argparse
import pandas as pd
import glob

sys.path.insert(0, '/uscms/home/benjtann/nobackup/ML/dihiggsMLProject/higgsReconstruction/')
from eventReconstructionClass import eventReconstruction

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--inputTXTFile", help="input txt file")
parser.add_argument("--outputTag", help="tag for output")

args = parser.parse_args()

if ( len(vars(args)) != 2 ): # 2 --> two for options 
    os.system('python fileHarvester.py -h')
    quit()

# ** A. Test input .txt file and exit if DNE
if(args.inputTXTFile is None):
    print ("#### Need to specify input .txt file using --inputTXTFile <address to .txt file> ####\nEXITING\n")
    quit()
else:
    if ( not os.path.exists(args.inputTXTFile) ):
        print ("#### Specified input file ({0}) DNE ####.\nEXITING\n".format(args.inputTXTFile))
        quit()
    else:
        print ('-- Setting inputTXTFile = {0}'.format(args.inputTXTFile))

# ** B. Test tag input and exit if DNE
if(args.outputTag is None):
    print ("#### Need to specify output tag using --outputTag <tag> ####\nEXITING\n")
    quit()
else:
    print ('-- Setting outputTag = {0}'.format(args.outputTag))


# *** 1. Create dummy instance of reco class
#eventReconstructor = eventReconstruction('dummy', 'dummy', True)
#eventReconstructor.setConsiderFirstNjetsInPT(4)
#eventReconstructor.setNJetsToStore(10)
#eventReconstructor.setRequireNTags(3)


# *** 2. Loop over files in .txt and run reconstruction
#passedFirstFile = False
for infile in open(args.inputTXTFile, 'r'):
    #if passedFirstFile is True:
    #    continue

    # ** A. Make some useful strings
    infile = infile.split('\n')[0]
    runSplit = infile.split('/')[-5]
    outName = args.outputTag + '_' + str(runSplit)
    workDir = '/'.join(infile.split('/')[:-4])
    isDihiggsSignal = True if 'pp2hh4b' in args.inputTXTFile else False


    os.system('cd {}'.format(workDir))

    # ** B. Run recontstruction
    eventReconstructor = eventReconstruction(outName, infile, isDihiggsSignal)
    eventReconstructor.setConsiderFirstNjetsInPT(4)
    eventReconstructor.setNJetsToStore(10)
    eventReconstructor.setRequireNTags(4)
    eventReconstructor.runReconstruction()

    print(workDir, outName)


    # ** C. Move output to runSplit directory for .csv organization
    os.system('cp -r {} {}'.format(outName, workDir))
    os.system('rm -rf {}'.format(outName))
    
    #passedFirstFile = True


# *** 3. Create combined .csv file?
print("######## Creating combined {} csv file".format(args.outputTag))
# ** A. Get list of all csv files
extension = 'csv'
sampleName = args.outputTag.split('__')[0]
all_filenames = [i for i in glob.glob('{}/*/{}*/*.{}'.format(sampleName, args.outputTag, extension))]

# ** B. Combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

# **C. Export to csv
combined_csv.to_csv( "{}_combined_csv.csv".format(args.outputTag), index=False, encoding='utf-8-sig')

