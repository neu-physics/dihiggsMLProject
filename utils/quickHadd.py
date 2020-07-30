##  Author:  Ben Tannenwald
##  Date:    July 30 2020
##  Purpose: quick script to hadd a bunch of files. not elegant, not efficient, but it works.

import os, argparse, glob

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputDir", help="input sample for getting files")
parser.add_argument("-n", "--nFilesPerMerge", help="number of input files in each merged file", type=int, default=10)

args = parser.parse_args()

if ( len(vars(args)) != 2 ): # --> depends on default options
    os.system('python quickHadd.py -h')
    print( vars(args), len(vars(args)))
    quit()

# *** 1. Test input directory existence and create output subdir
if(args.inputDir is None):
    print( "#### Need to specify output directory using --inputDir <desired input directory> ####\nEXITING")
    quit()
else:
    if ( not os.path.exists(args.inputDir) ):
        print( "Specified input directory ({0}) DNE.\nEXITING NOW".format(args.inputDir))
        quit()
    else:
        print( '-- Setting inputDir = {0}'.format(args.inputDir))
        os.system("mkdir {0}/summedOutputs/".format(args.inputDir))

if(args.nFilesPerMerge is None):
    print( "#### Need to specify nFilesPerMerge using --nFilesPerMerge <desired value> ####\nEXITING")
    quit()
else:
    print( '-- Setting nFilesPerMerge = {0}'.format(args.nFilesPerMerge))

# *** 2. Get original output files
splitFiles = glob.glob( '{}/*/pp*4b_14TeV_*PU_AUTO-v2/Events/run_01*/tag_1_delphes_events.root'.format(args.inputDir) )
print("Collected {} input files...".format( len(splitFiles) ))

# *** 3. How many input files do you want to merge in each output file? 
iMerged = 0
mergeFiles = []

for iFile in splitFiles:
    # ** A. if zero
    if len(mergeFiles)==0:
        mergeFiles.append(iFile)

    # ** B. add to merge list if below cutoff
    elif len(mergeFiles) < args.nFilesPerMerge:
        mergeFiles.append(iFile)

    # ** C. hadd if at user-specified cutoff
    elif len(mergeFiles)== args.nFilesPerMerge:
        haddString = "hadd {}_merge{}.root {}".format( args.inputDir, iMerged, ' '.join(mergeFiles) )
        #print( haddString)
        os.system( haddString )
        mergeFiles = []
        iMerged += 1
        print( "Merged {} file(s)...".format(iMerged) )

    # ** D. idk
    else:
        print("~~~~~~~ no idea what happened")






