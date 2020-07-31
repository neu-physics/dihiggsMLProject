from  eventReconstructionClass import eventReconstruction

#qcd_CMS_top4Tags_store8jets_1of5 = eventReconstruction('ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_1of5', '/eos/uscms/store/user/benjtann/upgrade/samples/ppTo4b_2MEvents_0PU_v2-05/ppTo4b_2MEvents_0PU_v2-05__1of5.root', False)
#qcd_CMS_top4Tags_store8jets_1of5.setConsiderFirstNjetsInPT(4)
#qcd_CMS_top4Tags_store8jets_1of5.setNJetsToStore(8)
#qcd_CMS_top4Tags_store8jets_1of5.setRequireNTags(4)
#qcd_CMS_top4Tags_store8jets_1of5.runReconstruction()

#qcd_CMS_top4Tags_store8jets_2of5 = eventReconstruction('ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_2of5', '/eos/uscms/store/user/benjtann/upgrade/samples/ppTo4b_2MEvents_0PU_v2-05/ppTo4b_2MEvents_0PU_v2-05__2of5.root', False)
#qcd_CMS_top4Tags_store8jets_2of5.setConsiderFirstNjetsInPT(4)
#qcd_CMS_top4Tags_store8jets_2of5.setNJetsToStore(8)
#qcd_CMS_top4Tags_store8jets_2of5.setRequireNTags(4)
#qcd_CMS_top4Tags_store8jets_2of5.runReconstruction()

#qcd_CMS_top4Tags_store8jets_3of5 = eventReconstruction('ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_3of5', '/eos/uscms/store/user/benjtann/upgrade/samples/ppTo4b_2MEvents_0PU_v2-05/ppTo4b_2MEvents_0PU_v2-05__3of5.root', False)
#qcd_CMS_top4Tags_store8jets_3of5.setConsiderFirstNjetsInPT(4)
#qcd_CMS_top4Tags_store8jets_3of5.setNJetsToStore(8)
#qcd_CMS_top4Tags_store8jets_3of5.setRequireNTags(4)
#qcd_CMS_top4Tags_store8jets_3of5.runReconstruction()

#qcd_CMS_top4Tags_store8jets_4of5 = eventReconstruction('ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_4of5', '/eos/uscms/store/user/benjtann/upgrade/samples/ppTo4b_2MEvents_0PU_v2-05/ppTo4b_2MEvents_0PU_v2-05__4of5.root', False)
#qcd_CMS_top4Tags_store8jets_4of5.setConsiderFirstNjetsInPT(4)
#qcd_CMS_top4Tags_store8jets_4of5.setNJetsToStore(10)
#qcd_CMS_top4Tags_store8jets_4of5.setRequireNTags(4)
#qcd_CMS_top4Tags_store8jets_4of5.runReconstruction()

#qcd_CMS_top4Tags_store8jets_5of5 = eventReconstruction('ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_5of5', '/eos/uscms/store/user/benjtann/upgrade/samples/ppTo4b_2MEvents_0PU_v2-05/ppTo4b_2MEvents_0PU_v2-05__5of5.root', False)
#qcd_CMS_top4Tags_store8jets_5of5.setConsiderFirstNjetsInPT(4)
#qcd_CMS_top4Tags_store8jets_5of5.setNJetsToStore(8)
#qcd_CMS_top4Tags_store8jets_5of5.setRequireNTags(4)
#qcd_CMS_top4Tags_store8jets_5of5.runReconstruction()

hh_top4Tags_addCons = eventReconstruction('test_pp2hh4b_0PU_1M_addCons_oneFile', '/eos/uscms/store/user/benjtann/upgrade/samples/pp2hh4b_1MEvents_addConstituents_0PU_v2-06_r2/0-5000/pp2hh4b_14TeV_0PU_AUTO-v2/Events/run_01_decayed_1/tag_1_delphes_events.root', True, True)
hh_top4Tags_addCons.setConsiderFirstNjetsInPT(4)
hh_top4Tags_addCons.setNJetsToStore(10)
hh_top4Tags_addCons.setRequireNTags(4)
hh_top4Tags_addCons.setSaveJetConstituents(True)
hh_top4Tags_addCons.runReconstruction()

