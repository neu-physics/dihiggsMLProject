from keras.regularizers import l2, l1
from pandas import read_csv
import AE_utils
import FF_NeuralNet_Utils

#Some hyperparameters
vars = ['deltaPhi(h1 jets)', 'deltaPhi(h1, h2)', 'deltaPhi(h2 jets)', 'deltaR(h1 jets)', 'deltaR(h1, h2)',
        'deltaR(h2 jets)', 'h1_mass', 'h1_pt', 'h2_mass', 'hh_mass', 'hh_pt', 'jet1_eta', 'jet1_mass', 'jet1_pt',
        'jet1_pz', 'jet2_eta', 'jet2_mass', 'jet2_pz', 'jet3_eta', 'jet3_pt', 'jet4_eta', 'jet4_pt', 'jet4_px',
        'jet4_py', 'nJets', 'recoJet1_eta', 'recoJet1_mass', 'recoJet1_pt', 'recoJet1_pz', 'recoJet2_eta',
        'recoJet2_pt', 'recoJet3_eta', 'recoJet3_mass', 'recoJet3_pz', 'recoJet4_eta', 'recoJet4_pt', 'scalarHT']

varlen = len(vars)
testing_fraction = 0.2
l1_reg = l1(0.0001)
l2_reg = l2(0.0001) #More harsh against anomalies

#Import files
dihiggs_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\higgs\closestDijetMassesToHiggs.csv" #equalDijetMass_sig.csv
qcd_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\qcd\closestDijetMassesToHiggs.csv" #equalDijetMass_sig.csv
dfhiggs = read_csv(dihiggs_file)
dfqcd = read_csv(qcd_file)

#Preprocess data
qcd_train_set, qcd_test_set, higgs_test_set, test_set = AE_utils.process(dfhiggs, dfqcd, vars, testing_fraction)

#Run model on training set
history, autoencoder, encoder  = AE_utils.AEmodel(qcd_train_set, varlen, l1_reg)
autoencoder.summary()
AE_utils.epoch_history(history)
qcdlosshistory, higgslosshistory, loss_threshold = AE_utils.AE_statistics(autoencoder, qcd_train_set,
                                                                 qcd_test_set, higgs_test_set)
AE_utils.loss_distribution(qcdlosshistory, higgslosshistory)
AE_utils.significacne(qcdlosshistory, higgslosshistory, loss_threshold, testing_fraction)
