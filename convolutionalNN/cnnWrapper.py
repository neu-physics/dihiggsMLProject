from cnnModelClass import cnnModelClass

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
