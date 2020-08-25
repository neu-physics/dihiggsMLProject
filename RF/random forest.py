from pandas import read_csv
import RandomForests_utils
from joblib import dump

tree_filepath = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\500kHiggs&2MQCDspreadsheets\ConsiderFirstNjetsInPT=4\RF plots\trees"

#Some hyperparameters
vars = ['deltaR(h1, h2)', 'h1_mass', 'h2_mass', 'deltaR(h1 jets)', 'scalarHT', 'h1_pt', 'deltaPhi(h1 jets)', 'hh_mass',
        'hh_pt', 'deltaR(h2 jets)', 'jet3_pt', 'jet4_pt', 'jet2_eta', 'jet3_eta', 'deltaPhi(h2 jets)', 'nJets', 'h2_pt']
varlen = len(vars)
testing_fraction = 0.3

dihiggs_filepath = r"C:\Users\Colby\Box Sync\Neu-work\delphes stuff\1MHiggs&4MQCDspreadsheets\higgs sheets\equalDijetMass_higgs.zip"
qcd_filepath = r"C:\Users\Colby\Box Sync\Neu-work\delphes stuff\1MHiggs&4MQCDspreadsheets\qcd sheets\equalDijetMass_qcd.csv"
modelcheckpoint_filepath = r"C:\Users\Colby\Box Sync\Neu-work\delphes stuff\1MHiggs&4MQCDspreadsheets\model files\RF\model.hdf5"

#Import files, convert them to dataframes, delte the files
dihiggs_file = RandomForests_utils.extractfiles(dihiggs_filepath)
dfhiggs = read_csv(dihiggs_file)
dfqcd = read_csv(qcd_filepath)
RandomForests_utils.deletefiles(dihiggs_file)

#Preprocess data
data_train, data_test, label_train, label_test = RandomForests_utils.process(dfhiggs, dfqcd, vars, testing_fraction,
                                                                                                 nJetsFunction=False, nJetstoConsider=False,
                                                                                                 nBTagsFunction=False, nBTagstoConsider=False,
                                                                                                equal_qcd_dihiggs_samples=True)
num_trees = 300

#Define parameters for RF
params = {
    'learning_rate': 1,
    'n_estimators': num_trees,
    'max_depth': 20,
    'min_child_weight': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 1.175,
    #'reg_lambda': 3,
    'scale_pos_weight': 0.2,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42

}

model, predictions = RandomForests_utils.trainBDT(data_train, label_train, data_test, label_test, params, min_background=400)
dump(model, modelcheckpoint_filepath)
RandomForests_utils.plot_feature_importances(model, vars)
RandomForests_utils.roc_plot(label_test, predictions)
