from imageMaker import imageMaker

#test = imageMaker('test_pp2hh4b_testSingle_3color_pix31_phiCM', 'test_pp2hh4b_pkl.txt', isSignal=True, testRun=False, addImpactParameters=False) # test only
#test = imageMaker('test_pp2hh4b_testSingle_5color_pix31_phiCM_allColors', 'test_pp2hh4b_pkl.txt', isSignal=True, testRun=False, addImpactParameters=True) # test only
test = imageMaker('test_ppTo4b_testSingle_5color_pix31_phiCM_allColors', 'test_ppTo4b_pkl.txt', isSignal=True, testRun=False, addImpactParameters=True) # test only
#test = imageMaker('test_pp2hh4b_testSingle_composite_pix15_phiCM_cat', 'pp2hh4b_singleList.txt', True) # test only
#test2 = imageMaker('test_ppTo4b_testSingle_composite_pix15_phiCM_cat', 'ppTo4b_singleList.txt', False) # test only
#test = imageMaker('test_newArch3_pp2hh4b', 'test_pp2hh4b_pkl.txt', True) # 25k dihiggs
#test2 = imageMaker('ppTo4b_4M_20files_composite_pix15_phiCM_cat', 'ppTo4b_4M_pickleList_20files.txt', False) # 50k qcd

test.processFiles()
#test2.processFiles()

