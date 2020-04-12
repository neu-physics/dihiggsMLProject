from sklearn import preprocessing
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from pandas import DataFrame

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


def AE_statistics(model, qcd_train, qcd_test, higgs_test, loss_threshold): #Finds loss threshold
    n = qcd_train.sample(n=len(qcd_test)) #Comparison set
    m = qcd_train.sample(n=len(higgs_test)) #Comparison set
    print(len(n), len(m))
    loss_history = LossHistory()

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    qcdloss = model.evaluate(qcd_test, n, batch_size=1, callbacks=[loss_history]) #Evaluate the model on background, gather TN & FP data
    qcdlosshistory = loss_history.losses
    for x in qcdlosshistory:
        if x > loss_threshold:
            false_positives.append(1)
        else:
            true_negatives.append(1)

    higgsloss = model.evaluate(higgs_test, m, batch_size=1, callbacks=[loss_history]) #Evaluate the model on signal, gather TP & FN data
    higgslosshistory = loss_history.losses
    for x in higgslosshistory:
        if x > loss_threshold:
            true_positives.append(1)
        else:
            false_negatives.append(1)

    #Gather mean loss for qcd and sig, along with other data
    midpoint = (abs(higgsloss-qcdloss))/2
    if qcdloss < higgsloss:
        loss_thresh = qcdloss+midpoint
    elif qcdloss > higgsloss:
        loss_thresh = higgsloss+midpoint
    else:
        loss_thresh = qcdloss
        print("qcdloss = higgsloss")

    qcdloss = str(round(qcdloss, 3))
    higgsloss = str(round(higgsloss, 3))
    loss_thresh = str(round(loss_thresh, 3))
    midpoint = str(round(midpoint, 3))

    print('QCDLoss = ' + qcdloss + '\nHiggsLoss = ' + higgsloss + '\nLossThreshold = ' + loss_thresh +
          '\nMidDifference = ' + midpoint)

    #Print statistics
    precision = len(true_positives)/(len(true_positives)+len(false_positives)) #Gives ratio of true positives predicted by model i.e. percentage of correct anomaly predictions
    recall = len(true_positives)/(len(m)) #Gives the ratio of anomalies detected
    print("Precision (Percentage of corrent anomaly predictions): {}%\nRecall (Percentage of anomalies detected): {}%"
          .format(precision*100, recall*100))


    #Plot loss distribution
    plt.figure("Loss distribution")
    plt.hist(qcdlosshistory, label='QCD', color='lightgrey', bins=100, alpha=0.4, density=True)
    plt.hist(higgslosshistory, label='Higgs', color='cyan', bins=100, alpha=0.4, density=True)
    plt.xlabel('Loss distribution')
    plt.legend()
    plt.show()
