from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from pandas import read_csv
from FF_NeuralNet_Utils import process, significance, accuracy, epoch_history
import time

#Some hyperparameters
testing_fraction = 0.2
dropout = 0.2
l2_reg = l2(0.0001)
vars = ['deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'hh_mass', 'jet2_eta', 'jet3_eta', 'nJets', 'jet3_pz',
        'jet4_eta', 'h1_mass', 'h2_mass','hh_pt', 'h1_pt', 'scalarHT', 'deltaPhi(h1, h2)', 'deltaPhi(h1 jets)', 'jet4_pt',
        'jet2_pt', 'jet1_pz', 'jet1_pt', 'jet1_mass', 'jet1_eta']

#Grab spreadsheets and parse through pandas
dihiggs_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\higgs\equalDijetMass_sig.csv"
qcd_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\qcd\equalDijetMass_bkg.csv"
dfhiggs = read_csv(dihiggs_file)
dfqcd = read_csv(qcd_file)

#Preprocess data
data_test, data_train, data_val, label_test, label_train, label_val = process(dfhiggs, dfqcd, vars, testing_fraction)

#Start train model timer
start_time = time.time()

#Create model
model = Sequential()
model.add(Dense(175, activation='relu', input_dim=len(vars), kernel_regularizer=l2_reg))
model.add(Dropout(dropout))
model.add(BatchNormalization())

model.add(Dense(90, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
#Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#Setup callbacks
fit_callbacks = [EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=10, min_delta=0.001, restore_best_weights=True)]
#Train model
history = model.fit(data_train, label_train, validation_data=(data_val, label_val), epochs=100, callbacks=fit_callbacks, verbose=2)

#Finish train model timer
model_time = time.time() - start_time

#Analyze model
trainscores, testscores = accuracy(model, data_train, label_train, data_test, label_test)
print(trainscores, '\n', testscores)
print("--- Train model time %s seconds ---" % model_time)
significance(model, data_test, label_test, testing_fraction)
epoch_history(history)
model.summary()
