import os, sys, argparse
from cnnModelClass import cnnModelClass

# *** 0. setup parser for command line
parser = argparse.ArgumentParser()
parser.add_argument("--outputDir", help="output directory for model training outputs")
parser.add_argument("--inputHHFile", help=".txt file containing input .h5 files for dihiggs")
parser.add_argument("--inputQCDFile", help=".txt file containing input .h5 files for qcd")
parser.add_argument('-x','--extraVariables', nargs='+', help='add extra variables to concatenate with convolutional outputs')
parser.add_argument('-i','--imageCollections', nargs='+', help='imageCollections to process', required=True)
parser.add_argument('--addClassWeights', dest='addClassWeights', action='store_true')
parser.add_argument('--testRun', dest='testRun', action='store_true')
parser.add_argument('--condorRun', dest='condorRun', action='store_true')
parser.set_defaults(addClassWeights=False)
parser.set_defaults(testRun=False)
parser.set_defaults(condorRun=False)

args = parser.parse_args()


if ( len(vars(args)) != 8 ): # --> depends on default options
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
    print( "#### No extra variables specified. Next time add variables (if desired) using --extraVariables <extra vars separated by spaces> ####\n")
    print( '-- Setting extraVariables = {0}'.format(args.extraVariables))
    args.extraVariables = []
else:
    print( '-- Setting extraVariables = {0}'.format(args.extraVariables))


modelArgs = dict(
    #3xconv, 2xPool
    #_cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (2, 2)]] ],
    #2xconv w/ 16 filters, 2xPool
    #_cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
    #2xconv w/ 32 filters, 2xPool
    #_cnnLayers= [ ['Conv2D',[32, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[32, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
    #2xconv w/ 16-32 filters, 2xPool
    #_cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[32, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
    #2xconv w/ 16-32 filters, 1xPool
    #_cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['Conv2D',[32, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
    #2xconv w/ 16-16 filters, 1xPool
    _cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
    #2xconv w/ 8-16 filters, 1xPool
    #_cnnLayers= [ ['Conv2D',[8, (3, 3)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],

    _ffnnLayers= [ ['Dense', [64]], ['BatchNormalization'], ['Dense', [64]] ],
    _loadSavedModel = False,
    _useClassWeights=args.addClassWeights,
    _extraVariables = args.extraVariables,
    #_extraVariables=['HT', 'nJets', 'nBTags'],
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

for iCollection in range(0, len(args.imageCollections)):
    imageCollection = args.imageCollections[iCollection]

    if iCollection == 0: # first model, create class
        cnn = cnnModelClass('{}_{}_{}'.format(imageCollection, weightsTag, percentTag), 
                            **classArgs,
                             _imageCollection = imageCollection,
                            _testRun = args.testRun,
                            _condorRun = args.condorRun
                            )

    else: # reinitialize to create new model
        cnn.reinitialize('{}_{}_{}'.format(imageCollection, weightsTag, percentTag), 
                         **modelArgs,
                         _imageCollection = imageCollection,
                         )

    cnn.run()

