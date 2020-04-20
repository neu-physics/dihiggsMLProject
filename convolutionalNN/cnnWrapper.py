from cnnModelClass import cnnModelClass

"""
test = cnnModelClass('test_cnnModelClass_r0',
                     _cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (2, 2)]] ],
                     _ffnnLayers= [ ['Dense', [64]], ['BatchNormalization'], ['Dense', [64]] ],
                     _hhFile = '/home/btannenw/Desktop/ML/dihiggsMLProject/convolutionalNN/pp2hh4b_1M_20files_composite_pix15_phiCM_cat/images/pp2hh4b_1M_20files_composite_pix15_phiCM_cat_allImages.h5',
                     _qcdFile = '/home/btannenw/Desktop/ML/dihiggsMLProject/convolutionalNN/ppTo4b_4M_20files_composite_pix15_phiCM_cat/images/ppTo4b_4M_20files_composite_pix15_phiCM_cat_allImages.h5',
                     _imageCollection = 'compositeImgs',
                     #_loadSavedModel = False,
                     _loadSavedModel = True,
                     _testRun = True,
)
test.run()
"""

# multi-run
imageCollections = ['compositeImgs','compositeImgs_<4j','compositeImgs_>=4j0b','compositeImgs_>=4j1b',
                    'compositeImgs_>=4j2b','compositeImgs_>=4j3b','compositeImgs_>=4j4b','compositeImgs_>=4j>=4b',
                    'trackImgs', 'nHadronImgs', 'photonImgs',
                    ]

modelArgs = dict(#3xconv, 2xPool
                 #_cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (2, 2)]] ],
                 #2xconv, 2xPool
                _cnnLayers= [ ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]], ['Conv2D',[16, (3, 3)]], ['MaxPooling2D', [(2,2)]] ],
                _ffnnLayers= [ ['Dense', [64]], ['BatchNormalization'], ['Dense', [64]] ],
                _hhFile = './imageList_pp2hh4b_test.txt',
                _qcdFile = './imageList_ppTo4b_test.txt',                
                _loadSavedModel = False,
)

for iCollection in imageCollections:   
    cnn = cnnModelClass('cnnModelClass_{}_2Conv_2MaxPool_2Dense_noWeights_10percent'.format(iCollection),
                        **modelArgs,
                        _imageCollection = iCollection,
                        #_testRun = True,
                        #_useClassWeights=False,
    )

    cnn_classWeights = cnnModelClass('cnnModelClass_{}_2Conv_2MaxPool_2Dense_addClassWeights_10percent'.format(iCollection),
                               **modelArgs,
                               _imageCollection = iCollection,
                               #_testRun = True,
                               _useClassWeights=True,
    )

    cnn_classWeights.run()

    
