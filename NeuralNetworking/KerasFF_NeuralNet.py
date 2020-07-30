from keras.regularizers import l2
from pandas import read_csv
import FF_NeuralNet_Utils
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

#Some hyperparameters
vars = ['deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'hh_mass', 'jet2_eta', 'jet3_eta', 'nJets', 'jet3_pz',
        'jet4_eta', 'h1_mass', 'h2_mass','hh_pt', 'h1_pt', 'scalarHT', 'deltaPhi(h1, h2)', 'deltaPhi(h1 jets)', 'jet4_pt',
        'jet2_pt', 'jet1_pz', 'jet1_pt', 'jet1_mass', 'jet1_eta']
varlen = len(vars)
testing_fraction = 0.2
dropout = 0.2
l2_reg = l2(0.0001)
dihiggs_filepath = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\500kHiggs&2MQCDspreadsheets\ConsiderFirstNjetsInPT=4\higgs\closestDijetMassesToHiggs_higgs.zip"
qcd_filepath = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\500kHiggs&2MQCDspreadsheets\ConsiderFirstNjetsInPT=4\qcd\closestDijetMassesToHiggs_qcd.zip"

#Import files, convert them to dataframes, delte the files
dihiggs_file, qcd_file = FF_NeuralNet_Utils.extractfiles(dihiggs_filepath, qcd_filepath, is_qcd_multiple_files=True)
dfhiggs = read_csv(dihiggs_file)
#dfqcd = read_csv(qcd_file)
dfqcd = FF_NeuralNet_Utils.appendQCDfilestogether(qcd_file)
FF_NeuralNet_Utils.deletefiles(dihiggs_file, qcd_file, is_qcd_multiple_files=True)

#Preprocess data
data_test, data_train, data_val, label_test, label_train, label_val = FF_NeuralNet_Utils.process(dfhiggs, dfqcd, vars, testing_fraction,
                                                                                                 nJetsFunction='>=', nJetstoConsider=4,
                                                                                                 nBTagsFunction='>=', nBTagstoConsider=4)

#Start train model timer
start_time = time.time()

#Create model
model = Sequential()
model.add(Dense(175, activation='relu', input_dim=varlen, kernel_regularizer=l2_reg))
model.add(Dropout(dropout))
model.add(BatchNormalization())
model.add(Dense(90, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Setup callbacks
fit_callbacks = [EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=10, min_delta=0.001,
                               restore_best_weights=True)]
# Train model
history = model.fit(data_train, label_train, validation_data=(data_val, label_val), epochs=100,
                    callbacks=fit_callbacks, verbose=2)


#Finish train model timer
model_time = time.time() - start_time

#Analyze model
trainscores, testscores = FF_NeuralNet_Utils.accuracy(model, data_train, label_train, data_test, label_test)
print(trainscores, '\n', testscores)
print("--- Train model time %s seconds ---" % model_time)
FF_NeuralNet_Utils.significance(model, data_test, label_test, testing_fraction)
FF_NeuralNet_Utils.epoch_history(history)
model.summary()
