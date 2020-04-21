import os, sys, argparse
from cnnModelClass import cnnModelClass

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--outputDir", help="output directory for model training outputs")
parser.add_argument("--imageCollection", help="image collection")
parser.add_argument("--inputHHFile", help=".txt file containing input .h5 files for dihiggs")
parser.add_argument("--inputQCDFile", help=".txt file containing input .h5 files for qcd")
parser.add_argument('--addClassWeights', dest='addClassWeights', action='store_true')
parser.add_argument('--testRun', dest='testRun', action='store_true')
parser.set_defaults(addClassWeights=False)
parser.set_defaults(addClassWeights=False)

args = parser.parse_args()


if ( len(vars(args)) != 6 ): # 4/5/6 --> depends on default options
    os.system('python cnnWrapper.py -h')
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


# ** C. Test image collction and exit if DNE
if(args.imageCollection is None):
    print( "#### Need to specify input .txt file using --imageCollection <image collection> ####\nEXITING\n")
    quit()
else:
    print( '-- Setting imageCollection = {0}'.format(args.imageCollection))


# multi-run
#imageCollections = ['compositeImgs','compositeImgs_<4j','compositeImgs_>=4j0b','compositeImgs_>=4j1b',
#                    'compositeImgs_>=4j2b','compositeImgs_>=4j3b','compositeImgs_>=4j4b','compositeImgs_>=4j>=4b',
#                    'trackImgs', 'nHadronImgs', 'photonImgs',
#]

modelArgs = dict(
    #3xconv, 2xPool
    #_cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (2, 2)]] ],
    #2xconv, 2xPool
    _cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
    _ffnnLayers= [ ['Dense', [64]], ['BatchNormalization'], ['Dense', [64]] ],
    _hhFile = args.inputHHFile, 
    _qcdFile = args.inputQCDFile, 
    _imageCollection = args.imageCollection,
    _loadSavedModel = False,
    #_datasetPercentage = 0.1,
    _datasetPercentage = 1.0,
)


cnn = cnnModelClass('cnnModelClass_{}_2Conv_2MaxPool_2Dense_noWeights_all'.format(args.imageCollection),
                    **modelArgs,
                    _testRun = args.testRun,
                    _useClassWeights=args.addClassWeights,
                )

cnn.run()

