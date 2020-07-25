import sys
sys.path.insert(0, '/uscms/home/benjtann/nobackup/ML/dihiggsMLProject/higgsReconstruction/')
#from eventReconstructionClass_Jet import eventReconstruction
from eventReconstructionClass import eventReconstruction

#root://cmseos.fnal.gov//eos/uscms/store/user/ali/diHiggs/pp2hh4b_14TeV_0PU_AUTO-v2_delphesV2_vanilla_v0.root
#file_path = "root://cmseos.fnal.gov//eos/uscms/store/user/benjtann/upgrade/samples/pp2hh4b_1MEvents_addConstituents_0PU_v2-06/*/pp2hh4b_14TeV_0PU_AUTO-v2/Events/run_01_decayed_1/tag_1_delphes_events.root"
file_path = 'root://cmseos.fnal.gov//eos/uscms/store/user/benjtann/upgrade/samples/pp2hh4b_1MEvents_addConstituents_0PU_v2-06/147500-150000/pp2hh4b_14TeV_0PU_AUTO-v2/Events/run_01_decayed_1/tag_1_delphes_events.root'
#file_path = 'root://cmseos.fnal.gov//eos/uscms/store/user/benjtann/upgrade/samples/pp2hh4b_1MEvents_addConstituents_0PU_v2-06/15000-17500/pp2hh4b_14TeV_0PU_AUTO-v2/Events/run_01_decayed_1/tag_1_delphes_events.root'
reconstructor = eventReconstruction('test2_JetReco_6', file_path, True, _isTestRun = False)

reconstructor.setConsiderFirstNjetsInPT(4)
reconstructor.setNJetsToStore(10)
reconstructor.setRequireNTags(2)
reconstructor.setSaveJetConstituents(True)
reconstructor.runReconstruction()

