from cnnModelClass import cnnModelClass

test = cnnModelClass('test_cnnModelClass_r0',
                     _cnnLayers= [ [], [], [], []],
                     _ffnnLayers= [],
                     _hhFile = 'sig.h5',
                     _qcdFile = 'qcd.h5',
                     _imageCollection = 'compositeImgs',
                     _loadSavedModel = False,
                     _testRun = True
)

test.run()
