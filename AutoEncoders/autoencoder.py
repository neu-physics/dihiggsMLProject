from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from pandas import read_csv
from AE_utils import process, epoch_history, significance
import time

#Some hyperparameters
vars = ['deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'hh_mass', 'jet2_eta', 'jet3_eta', 'nJets', 'jet3_pz',
        'jet4_eta', 'h1_mass', 'h2_mass','hh_pt', 'h1_pt', 'scalarHT', 'deltaPhi(h1, h2)', 'deltaPhi(h1 jets)', 'jet4_pt',
        'jet2_pt', 'jet1_pz', 'jet1_pt', 'jet1_mass', 'jet1_eta']
varlen = len(vars)
encoding_dim = int(varlen/2)
latent_dim = int(encoding_dim/2)
testing_fraction = 0.2

#Import files
dihiggs_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\higgs\equalDijetMass_sig.csv"
qcd_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\qcd\equalDijetMass_bkg.csv"
dfhiggs = read_csv(dihiggs_file)
dfqcd = read_csv(qcd_file)

#Preprocess data
qcd_set, dihiggs_set = process(dfhiggs, dfqcd, vars)

#Design model
model = Sequential()
model.add(Dense(units=encoding_dim, input_dim=varlen, activation='relu')) #Input + Encoding layer 1
#model.add(Dense(units=35, activation='relu')) #Encoding layer 2
#model.add(Dense(units=35, activation='relu')) #Decoding layer 1
#model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=varlen, activation='sigmoid')) #Decoding layer/Output
#model.add(Dense(units=varlen, activation='sigmoid')) #Output layer

#Compile and fit model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(qcd_set, qcd_set, epochs=10)
model.summary()

#Show statistics of model
significance(qcd_set, dihiggs_set, testing_fraction, vars, model)
epoch_history(history)

