from sklearn import preprocessing
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from statistics import mean, stdev
from commonFunctions import returnBestCutValue, compareManyHistograms
import numpy as np
import scipy.stats as st

class LossHistory(Callback):
    def on_test_begin(self, logs=None):
        self.losses = []

    def on_test_batch_end(self, batch, logs=None):
        self.losses.append(logs['loss'])


def process(dfhiggs, dfqcd, vars, testing_fraction):
# Sort top variables and scale both dataframes to same scale
    scaler = preprocessing.StandardScaler()
    scaler.fit(dfqcd[vars])
    qcdtopvars = scaler.transform(dfqcd[vars])
    higgstopvars = scaler.transform(dfhiggs[vars])

#Train test split
    qcd_train_set, qcd_test_set = train_test_split(qcdtopvars, test_size=testing_fraction, shuffle=True)
    higgs_train_set, higgs_test_set = train_test_split(higgstopvars, test_size=testing_fraction, shuffle=True)
    qcd_train_set = DataFrame(qcd_train_set)
    qcd_test_set = DataFrame(qcd_test_set)
    higgs_test_set = DataFrame(higgs_test_set)
    test_set = qcd_test_set.append(higgs_test_set)
    test_set.sample(frac=1)

    return qcd_train_set, qcd_test_set, higgs_test_set, test_set


def AEmodel(input_set, varlen, regularizer):
    # this is our input placeholder
    input_img = Input(shape=(varlen,))
    #encoded representation of the input
    encoded = Dense(units=5, activation='relu', activity_regularizer=regularizer)(input_img)
    #the lossy reconstruction of the input
    decoded = Dense(varlen, activation='sigmoid')(encoded)

    #Full autoencoder
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    #Compile and fit autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # Set up callbacks
    fit_callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=0.001, patience=10, restore_best_weights=True)]
                    #ModelCheckpoint(filepath=filepath, monitor='loss', verbose=2, save_best_only=True)]
    history = autoencoder.fit(input_set, input_set, epochs=100, shuffle=True, verbose=2, callbacks=fit_callbacks,
                              batch_size=1024)

    return history, autoencoder, encoder


def epoch_history(history):
#Summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('AE Model Loss')
    plt.ylabel('Loss [A.U.]')
    plt.xlabel('Epoch')
    plt.show()


def reject_outliers(data, m = 4): #Used to remove outliers from a list, the smaller m is the more harsh the rejection
    array = np.array(data)
    d = np.abs(array - np.median(array))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return array[s<m]


def AE_statistics(model, qcd_train, qcd_test, higgs_test): #Finds loss threshold
    n = qcd_train.sample(n=len(qcd_test)) #Comparison set
    m = qcd_train.sample(n=len(higgs_test)) #Comparison set
    loss_history = LossHistory()

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    model.evaluate(qcd_test, n, batch_size=1, callbacks=[loss_history]) #Evaluate the model on background
    qcdlosshistory_raw = loss_history.losses
    qcdlosshistory = reject_outliers(qcdlosshistory_raw)
    qcdlossmean = mean(qcdlosshistory)

    model.evaluate(higgs_test, m, batch_size=1, callbacks=[loss_history]) #Evaluate the model on signal
    higgslosshistory_raw = loss_history.losses
    higgslosshistory = reject_outliers(higgslosshistory_raw)
    higgslossmean = mean(higgslosshistory)

    #Calculate loss threshold and mean difference
    meandifference = abs(higgslossmean-qcdlossmean)
    bigset = np.append(qcdlosshistory, higgslosshistory)
    loss_threshold = mean(bigset)

    #Append TP/FP data
    for x in qcdlosshistory:
        if x > loss_threshold:
            false_positives.append(1)
        else:
            true_negatives.append(1)
    for x in higgslosshistory:
        if x > loss_threshold:
            true_positives.append(1)
        else:
            false_negatives.append(1)

    qcdlossmean = str(round(qcdlossmean, 3))
    higgslossmean = str(round(higgslossmean, 3))
    loss_threshold = str(round(loss_threshold, 3))
    meandifference = str(round(meandifference, 3))

    print('QCDLoss = ' + qcdlossmean + '\nHiggsLoss = ' + higgslossmean + '\nLossThreshold = ' + loss_threshold +
          '\nMeanDifference = ' + meandifference)

    #Print statistics
    precision = len(true_positives)/(len(true_positives)+len(false_positives)) #Gives ratio of true positives predicted by model i.e. percentage of correct anomaly predictions
    recall = len(true_positives)/(len(higgslosshistory)) #Gives the ratio of anomalies detected
    print("Precision (Percentage of corrent anomaly predictions): {}%\nRecall (Percentage of anomalies detected): {}%"
          .format(precision*100, recall*100))

    return qcdlosshistory, higgslosshistory, loss_threshold


def loss_distribution(qcdlosshistory, higgslosshistory): #Plot loss distribution
    plt.figure("Loss distribution")
    plt.hist(qcdlosshistory, label='QCD', color='lightgrey', bins=100, alpha=0.4, density=True)
    plt.hist(higgslosshistory, label='Higgs', color='cyan', bins=100, alpha=0.4, density=True)
    plt.xlabel('Loss distribution')
    plt.legend()
    plt.show()

def significacne(qcdlosshistory, higgslosshistory, loss_threshold, testing_fraction): #Cacluates the significance
    loss_threshold = float(loss_threshold)
    #Combine sets with labels
    qcddf = DataFrame(qcdlosshistory)
    higgsdf = DataFrame(higgslosshistory)
    qcddf['Signal'] = 0
    higgsdf['Signal'] = 1

    # Calculate standard deviation of combined set
    bigset = higgsdf.append(qcddf)
    stdeviation = stdev(bigset[0])
    pred_hh = []
    pred_qcd = []

    #Calculate and append the predictions
    for index, row in bigset.iterrows():
        z = (row[0] - loss_threshold)/stdeviation
        if row['Signal'] == 0:
            pred_qcd.append(st.norm.cdf(z))
        else:
            pred_hh.append(st.norm.cdf(z))

    #Plot prediction histograms and show significance
    _nBins = 40
    predictionResults = {'hh_pred': pred_hh, 'qcd_pred': pred_qcd}
    compareManyHistograms(predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'ff-NN Score', 0, 1,
                          _nBins, _yMax=5, _normed=True, _savePlot=False)
    # Show significance
    returnBestCutValue('ff-NN', pred_hh.copy(), pred_qcd.copy(), _minBackground=400e3,
                       _testingFraction=testing_fraction)
