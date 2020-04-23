import os, sys, argparse
from cnnModelClass import cnnModelClass

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--outputDir", help="output directory for model training outputs")
#parser.add_argument("--imageCollection", help="image collection")
parser.add_argument("--inputHHFile", help=".txt file containing input .h5 files for dihiggs")
parser.add_argument("--inputQCDFile", help=".txt file containing input .h5 files for qcd")
parser.add_argument('--addClassWeights', dest='addClassWeights', action='store_true')
parser.add_argument('--testRun', dest='testRun', action='store_true')
parser.set_defaults(addClassWeights=False)
parser.set_defaults(testRun=False)

args = parser.parse_args()


if ( len(vars(args)) != 5 ): # 4/5/6 --> depends on default options
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
#if(args.imageCollection is None):
#    print( "#### Need to specify input .txt file using --imageCollection <image collection> ####\nEXITING\n")
#    quit()
#else:
#    print( '-- Setting imageCollection = {0}'.format(args.imageCollection))


# multi-run
imageCollections = ['compositeImgs','compositeImgs_lessThan4j',#'compositeImgs_ge4jInclb','compositeImgs_ge4j0b','compositeImgs_ge4j1b',
                    #'compositeImgs_ge4j2b','compositeImgs_ge4j3b','compositeImgs_ge4j4b','compositeImgs_ge4jge4b', 
                    #'compositeImgs_HT150','compositeImgs_HT300', 'compositeImgs_HT450',
                    #'trackImgs', 'nHadronImgs', 'photonImgs',
]



modelArgs = dict(
    #3xconv, 2xPool
    #_cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (2, 2)]] ],
    #2xconv, 2xPool
    _cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
    _ffnnLayers= [ ['Dense', [64]], ['BatchNormalization'], ['Dense', [64]] ],
    _loadSavedModel = False,
    _useClassWeights=args.addClassWeights,
)

classArgs = modelArgs.copy()
classArgs['_topDir'] = args.outputDir
classArgs['_hhFile'] = args.inputHHFile
classArgs['_qcdFile'] = args.inputQCDFile
#classArgs['_datasetPercentage'] = 0.1
#classArgs['_datasetPercentage'] = 0.2
classArgs['_datasetPercentage'] = 0.8

weightsTag = 'addClassWeights' if args.addClassWeights else 'noWeights'
percentTag = '80percent'

for iCollection in range(0, len(imageCollections)):
    imageCollection = imageCollections[iCollection]

    if iCollection == 0: # first model, create class
        cnn = cnnModelClass('cnnModelClass_{}_2Conv_2MaxPool_2Dense_{}_{}'.format(imageCollection, weightsTag, percentTag),
                            **classArgs,
                            _imageCollection = imageCollection,
                            _testRun = args.testRun
        )

    else: # reinitialize to create new model
        cnn.reinitialize('cnnModelClass_{}_2Conv_2MaxPool_2Dense_{}_{}'.format(imageCollection, weightsTag, percentTag),
                         **modelArgs,
                         _imageCollection = imageCollection,
                     )

    cnn.run()

