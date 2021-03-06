from keras.regularizers import l2, l1
from pandas import read_csv
import AE_utils
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint

#Some hyperparameters
vars = ['deltaPhi(h1 jets)', 'deltaR(h1 jets)', 'deltaR(h1, h2)',
        'deltaR(h2 jets)', 'h1_mass', 'h1_pt', 'h2_mass', 'recoJet4_pt', 'recoJet2_pt', 'scalarHT', 'hh_pt']
varlen = len(vars)
testing_fraction = 0.2
l1_reg = l1(0.0001)
l2_reg = l2(0.0001) #More harsh against anomalies
dihiggs_filepath = r"C:\Users\Colby\Box Sync\Neu-work\delphes stuff\1MHiggs&4MQCDspreadsheets\higgs sheets\equalDijetMass_higgs.zip"
qcd_filepath = r"C:\Users\Colby\Box Sync\Neu-work\delphes stuff\1MHiggs&4MQCDspreadsheets\qcd sheets\equalDijetMass_qcd.csv"
modelcheckpoint_filepath = r"C:\Users\Colby\Box Sync\Neu-work\delphes stuff\1MHiggs&4MQCDspreadsheets\model files\AE\bestmodel.hdf5"

#Import files, convert them to dataframes, delte the files
dihiggs_file = AE_utils.extractfiles(dihiggs_filepath)
dfhiggs = read_csv(dihiggs_file)
dfqcd = read_csv(qcd_filepath)
AE_utils.deletefiles(dihiggs_file)


#Preprocess data
qcd_train_set, qcd_val_set, qcd_test_set, higgs_train_set, higgs_val_set, higgs_test_set = \
                                                                AE_utils.process(dfhiggs, dfqcd, vars, testing_fraction,
                                                                 nJetsFunction=False, nJetstoConsider=False,
                                                                 nBTagsFunction=False, nBTagstoConsider=False,
                                                                 contamination_fraction=False, equal_qcd_dihiggs_samples=True)


#Run model on training set
input_img = Input(shape=(varlen,))  # Input
# hiddenlayer1 = Dense(units=7, activation='relu', activity_regularizer=regularizer)(input_img)
encoded = Dense(units=3, activation='relu', activity_regularizer=l2_reg)(input_img)  # Latent Layer
# hiddenlayer2 = Dense(units=7, activation='relu')(encoded)
decoded = Dense(varlen, activation='sigmoid')(encoded)  # Output

# Compile and fit autoencoder
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Set up callbacks
fit_callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=0.001, patience=50, restore_best_weights=True),
                 ModelCheckpoint(filepath=modelcheckpoint_filepath, monitor='loss', mode='min', verbose=0, save_best_only=True)]
history = autoencoder.fit(qcd_train_set, qcd_train_set, validation_data=(qcd_val_set, qcd_val_set), epochs=10000, shuffle=True, verbose=2, callbacks=fit_callbacks,
                          batch_size=1024)


autoencoder.summary()
AE_utils.epoch_history(history)
qcdlosshistory, higgslosshistory = AE_utils.AE_statistics(autoencoder, qcd_train_set, qcd_test_set, higgs_test_set, logarithmic=False)
#qcdlosshistory = AE_utils.get_outliers(qcdlosshistory, m=5, keep_outliers=1)
#higgslosshistory = AE_utils.get_outliers(higgslosshistory, m=5, keep_outliers=1)
AE_utils.significacne(qcdlosshistory, higgslosshistory, testing_fraction)
AE_utils.loss_plot(qcdlosshistory, higgslosshistory, remove_Outliers=True)
AE_utils.roc_plot(qcdlosshistory, higgslosshistory)
