from sklearn import preprocessing
from matplotlib import pyplot as plt
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pandas import DataFrame, read_csv, concat
from commonFunctions import returnBestCutValue, compareManyHistograms
from statistics import mean
import numpy as np
from zipfile import ZipFile
import os
import sys

class LossHistory(Callback):
    def on_test_begin(self, logs=None):
        self.losses = []

    def on_test_batch_end(self, batch, logs=None):
        self.losses.append(logs['loss'])


def extractfiles(dihiggs_filepath, qcd_filepath):
#Extracts csv data from a zip folder assuming qcd and dihiggs are same reco algorithm, also assuming qcd is split into 5 csvs
    with ZipFile(dihiggs_filepath, 'r') as zip:
        zip.extractall()
        dihiggs_file = zip.namelist()[0]
    with ZipFile(qcd_filepath, 'r') as zip:
        zip.extractall()
        qcd_file = zip.namelist()

    return dihiggs_file, qcd_file


def deletefiles(dihiggs_file, qcd_file):
#Deletes files from current directory
    if os.path.exists(dihiggs_file):
        os.remove(dihiggs_file)
    else:
        print("Cannot delete file, it does not exist")
    for file in qcd_file:
        if os.path.exists(file):
            os.remove(file)
        else:
            print("Cannot delete file, it does not exist")


def appendQCDfilestogether(qcd_files): #Assuming 5 qcd files
    a = read_csv(qcd_files[0])
    b = a.append(read_csv(qcd_files[1]))
    c = b.append(read_csv(qcd_files[2]))
    d = c.append(read_csv(qcd_files[3]))
    dfqcd = d.append(read_csv(qcd_files[4]))

    return dfqcd


def fixNJets(dfhiggs, dfqcd, jetstoconsider, function): #Removes rows in the dataset whose nJets > jetstoremove
    print("Dataframe shapes before removing NJets:\nDihiggs = " + str(dfhiggs.shape) + "\nQCD = " + str(dfqcd.shape))
    if function == '>':
        dfhiggsjetted = dfhiggs[dfhiggs.nJets > jetstoconsider]
        dfqcdjetted = dfqcd[dfqcd.nJets > jetstoconsider]
        print("nJets > " + str(jetstoconsider))
    elif function == '<':
        dfhiggsjetted = dfhiggs[dfhiggs.nJets < jetstoconsider]
        dfqcdjetted = dfqcd[dfqcd.nJets < jetstoconsider]
        print("nJets < " + str(jetstoconsider))
    elif function == '>=':
        dfhiggsjetted = dfhiggs[dfhiggs.nJets >= jetstoconsider]
        dfqcdjetted = dfqcd[dfqcd.nJets >= jetstoconsider]
        print("nJets >= " + str(jetstoconsider))
    elif function == '<=':
        dfhiggsjetted = dfhiggs[dfhiggs.nJets <= jetstoconsider]
        dfqcdjetted = dfqcd[dfqcd.nJets <= jetstoconsider]
        print("nJets <= " + str(jetstoconsider))
    elif function == '==':
        dfhiggsjetted = dfhiggs[dfhiggs.nJets == jetstoconsider]
        dfqcdjetted = dfqcd[dfqcd.nJets == jetstoconsider]
        print("nJets == " + str(jetstoconsider))
    else:
        print("Error, function must be defined as one of the following: '<', '>', '<=', '>=', '=='")
        sys.exit()
    print("Dataframe shapes after removing NJets:\nDihiggs = " + str(dfhiggsjetted.shape) + "\nQCD = " + str(dfqcdjetted.shape) +
    "\n==============================\n")

    return dfhiggsjetted, dfqcdjetted


def fixBTags(dfhiggs, dfqcd, tagstoconsider, function): #Removes rows in the dataset whose Btags > jetstoremove
    print("Dataframe shapes before removing BTags:\nDihiggs = " + str(dfhiggs.shape) + "\nQCD = " + str(dfqcd.shape))
    if function == '>':
        dfhiggstagged = dfhiggs[dfhiggs.nBTags > tagstoconsider]
        dfqcdtagged = dfqcd[dfqcd.nBTags > tagstoconsider]
        print("nBTags > " + str(tagstoconsider))
    elif function == '<':
        dfhiggstagged = dfhiggs[dfhiggs.nBTags < tagstoconsider]
        dfqcdtagged = dfqcd[dfqcd.nBTags < tagstoconsider]
        print("nBTags < " + str(tagstoconsider))
    elif function == '>=':
        dfhiggstagged = dfhiggs[dfhiggs.nBTags >= tagstoconsider]
        dfqcdtagged = dfqcd[dfqcd.nBTags >= tagstoconsider]
        print("nBTags >= " + str(tagstoconsider))
    elif function == '<=':
        dfhiggstagged = dfhiggs[dfhiggs.nBTags <= tagstoconsider]
        dfqcdtagged = dfqcd[dfqcd.nBTags <= tagstoconsider]
        print("nBTags <= " + str(tagstoconsider))
    elif function == '==':
        dfhiggstagged = dfhiggs[dfhiggs.nBTags == tagstoconsider]
        dfqcdtagged = dfqcd[dfqcd.nBTags == tagstoconsider]
        print("nBTags == " + str(tagstoconsider))
    else:
        print("Error, function must be defined as one of the following: '<', '>', '<=', '>=', '=='")
        sys.exit()
    print("Dataframe shapes after removing nBTags:\nDihiggs = " + str(dfhiggstagged.shape) + "\nQCD = " + str(dfqcdtagged.shape) +
    "\n==============================\n")

    return dfhiggstagged, dfqcdtagged


def contaminate(dihiggs_train, dihiggs_val, qcd_train, qcd_val, fraction): #Contaminates the qcd training set with a certain % of dihiggs signals
    n = len(qcd_train.sample(frac=fraction))
    if n <= len(dihiggs_train):
        sample = 1 - fraction
        dfqcd = qcd_train.sample(frac=sample)
        dfhiggs = dihiggs_train.sample(n=n)
        qcd_train_contaminated = concat([dfqcd, dfhiggs], ignore_index=True)
        print("QCD training set was successfully contaminated with dihiggs samples. Training set shape = " + str(qcd_train_contaminated.shape) +
              " with " + str(len(dfqcd)) + " QCD samples and " + str(len(dfhiggs)) + " dihiggs samples.")
    else:
        print("Error: Training contamination fraction was too large during preprocessing, not enough dihiggs samples to contaminate"
              "at that fraction. Try a smaller fraction.")
        sys.exit()
    m = len(qcd_val.sample(frac=fraction))
    if m <= len(dihiggs_val):
        sample = 1 - fraction
        dfqcdval = qcd_val.sample(frac=sample)
        dfhiggsval = dihiggs_val.sample(n=m)
        qcd_val_contaminated = concat([dfqcdval, dfhiggsval], ignore_index=True)
        print("QCD validation set was successfully contaminated with dihiggs samples. Validation set shape = " + str(qcd_val_contaminated.shape) +
              " with " + str(len(dfqcdval)) + " QCD samples and " + str(len(dfhiggsval)) + " dihiggs samples.")
    else:
        print("Error: Validation contamination fraction was too large during preprocessing, not enough dihiggs samples to contaminate"
            "at that fraction. Try a smaller fraction.")
        sys.exit()

    return qcd_train_contaminated, qcd_val_contaminated


def process(dfhiggs, dfqcd, vars, testing_fraction, nJetsFunction=False, nJetstoConsider=False, nBTagsFunction=False,
            nBTagstoConsider=False, contamination_fraction=False):
    print("Dataframe shapes before preprocessing:\nDihiggs = " + str(dfhiggs.shape) + "\nQCD = " + str(dfqcd.shape) +
    "\n==============================\n")
#If removeNJets is False, does not change the csv, otherwise it removes all jets > 'removeNJets'
    if nJetsFunction == False or nJetstoConsider == False:
        higgs_set_jetted = dfhiggs
        qcd_set_jetted = dfqcd
    else:
        higgs_set_jetted, qcd_set_jetted = fixNJets(dfhiggs, dfqcd, nJetstoConsider, nJetsFunction)

# If removeBTags is False, does not change the csv, otherwise it removes all jets > 'removeBTags'
    if nBTagsFunction == False or nBTagstoConsider == False:
        higgs_set = higgs_set_jetted
        qcd_set = qcd_set_jetted
    else:
        higgs_set, qcd_set = fixBTags(higgs_set_jetted, qcd_set_jetted, nBTagstoConsider, nBTagsFunction)

# Sort top variables and scale both dataframes to same scale
    scaler = preprocessing.StandardScaler()
    scaler.fit(qcd_set[vars])
    qcdtopvars = scaler.transform(qcd_set[vars])
    higgstopvars = scaler.transform(higgs_set[vars])

#Train test split
    qcd_train_set, qcd_test_set = train_test_split(qcdtopvars, test_size=testing_fraction, shuffle=True)
    higgs_train_set, higgs_test_set = train_test_split(higgstopvars, test_size=testing_fraction, shuffle=True)
    qcd_train_set, qcd_val_set = train_test_split(qcd_train_set, test_size=testing_fraction, shuffle=True)
    higgs_train_set, higgs_val_set = train_test_split(higgs_train_set, test_size=testing_fraction, shuffle=True)
    qcd_train_set = DataFrame(qcd_train_set)
    qcd_val_set = DataFrame(qcd_val_set)
    qcd_test_set = DataFrame(qcd_test_set)
    higgs_train_set = DataFrame(higgs_train_set)
    higgs_val_set = DataFrame(higgs_val_set)
    higgs_test_set = DataFrame(higgs_test_set)

    if contamination_fraction != False:
        qcd_train_set, qcd_val_set = contaminate(higgs_train_set, higgs_val_set, qcd_train_set, qcd_val_set, contamination_fraction)

    return qcd_train_set, qcd_val_set, qcd_test_set, higgs_train_set, higgs_val_set, higgs_test_set


def epoch_history(history):
#Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('AE Model Loss')
    plt.ylabel('Loss [A.U.]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def AE_statistics(model, qcd_train, qcd_test, higgs_test, logarithmic=False): #Get loss distributions
    if len(qcd_train) >= len(higgs_test):
        n = qcd_train.sample(n=len(qcd_test)) #Comparison set
        m = qcd_train.sample(n=len(higgs_test)) #Comparison set
        loss_history = LossHistory()

        model.evaluate(qcd_test, n, batch_size=1, callbacks=[loss_history])  # Evaluate the model on background
        qcdlosshistory = loss_history.losses

        model.evaluate(higgs_test, m, batch_size=1, callbacks=[loss_history])  # Evaluate the model on signal
        higgslosshistory = loss_history.losses
    else:
        n = qcd_train.sample(n=len(qcd_test))
        m = higgs_test.sample(n=len(qcd_train))
        loss_history = LossHistory()

        model.evaluate(qcd_test, n, batch_size=1, callbacks=[loss_history])  # Evaluate the model on background
        qcdlosshistory = loss_history.losses

        model.evaluate(m, qcd_train, batch_size=1, callbacks=[loss_history])  # Evaluate the model on signal
        higgslosshistory = loss_history.losses

    #Converts losshistory lists to log scale
    if logarithmic == True:
        qcdlosshistory = np.log(qcdlosshistory)
        higgslosshistory = np.log(higgslosshistory)
        print("Converted loss values to logarithmic numbers")
    else:
        pass

    return qcdlosshistory, higgslosshistory


def get_outliers(input_set, m=5, keep_outliers=0):
    data = np.array(input_set)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.

    if keep_outliers == 0:
        print("Removed outliers from dataset, shape = " + str(data[s < m].shape))
        return data[s < m]
    elif keep_outliers == 1:
        print("Removed values that weren't outliers from dataset, shape = " + str(data[s > m].shape))
        return data[s > m]
    else:
        print('Error: keep_outliers should = 0 or 1. 0 to return values < outlier, 1 to return values > outlier.')
        sys.exit()


def Pca(dfhiggs, vars):
    varlen = len(vars)
    pca = PCA(n_components=varlen)
    pca.fit_transform(dfhiggs[vars])

    features = range(pca.n_components)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.title('PCA for Dihiggs DF with ' + str(varlen) + ' variables')
    plt.xticks(features)
    plt.show()


def significacne(qcdlosshistory, higgslosshistory, testing_fraction): #Plot prediction histograms and show significance
    _nBins = 100
    predictionResults = {'hh_pred': higgslosshistory, 'qcd_pred': qcdlosshistory}
    compareManyHistograms(predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'AE score', 0, 3,
                          _nBins, _yMax=5, _normed=True, _savePlot=False)
    # Show significance
    returnBestCutValue('AE', higgslosshistory.copy(), qcdlosshistory.copy(), _minBackground=400e3,
                       _testingFraction=testing_fraction)

def loss_plot(qcdlosshistory, higgslosshistory, remove_Outliers=True):
    #Remove outliers to fix the graph
    if remove_Outliers == True:
        qcdlossnew = get_outliers(qcdlosshistory)
        higgslossnew = get_outliers(higgslosshistory)
    else:
        qcdlossnew = qcdlosshistory
        higgslossnew = higgslosshistory
    plt.hist(qcdlossnew, color='orange', alpha=0.4, density=True, label='Background', bins=100)
    plt.hist(higgslossnew, color='cornflowerblue', alpha=0.4, density=True, label='Signal', bins=100)
    plt.xlabel('Loss')
    plt.ylabel('Proportion')
    plt.suptitle('QCD vs Signal Loss Plot', fontweight='bold')
    plt.title("Mean QCD loss = " + str(round(mean(qcdlosshistory), 2)) + " Mean Dihiggs loss = " + str(round(mean(higgslosshistory), 2)))
    plt.legend()
    plt.show()


def average_loss_vs_Latentdimensions(input_set, varlen, regularizer, qcd_test, higgs_test, saveplots):
    latentsize = 1
    qcdloss = []
    higgsloss = []

    while latentsize <= varlen:
        history, autoencoder, encoder = AEmodel(input_set, varlen, regularizer, latentsize)
        qcdlosshistory, higgslosshistory = AE_statistics(autoencoder, input_set, qcd_test, higgs_test)
        qcdloss.append(mean(qcdlosshistory))
        higgsloss.append(mean(higgslosshistory))

        #Plot loss plot
        qcdlossnew = remove_outliers(qcdlosshistory)
        higgslossnew = remove_outliers(higgslosshistory)

        plt.hist(qcdlossnew, color='orange', alpha=0.4, density=True, label='Background', bins=100)
        plt.hist(higgslossnew, color='cornflowerblue', alpha=0.4, density=True, label='Signal', bins=100)
        plt.xlabel('Loss')
        plt.ylabel('Proportion')
        plt.suptitle('QCD vs Signal Loss Plot (Latent Dimension size = ' + str(latentsize) + ')', fontsize=14, fontweight='bold')
        plt.title('Mean QCD Loss = ' + str(round(mean(qcdlosshistory), 3)) + ', Mean Signal Loss = ' + str(round(mean(higgslosshistory), 3)))
        plt.legend()
        if saveplots == True:
            plt.savefig(r"C:\Users\Colby\Desktop\Neu-work\AE_notes\plots\LossDist(11 vars, latent=" + str(latentsize) + ".png")
            plt.close()
        else:
            plt.show()
        latentsize+=1

    #Plot average loss vs dimension
    plt.plot(qcdloss, marker='.', color='orange', label='QCD')
    plt.plot(higgsloss, marker='.', color='cornflowerblue', label='Signal')
    plt.xlabel('Latent Dimension Size')
    plt.ylabel('Average Loss')
    plt.title('Average Loss vs. Latent Dimension Size')
    plt.legend()
    if saveplots == True:
        plt.savefig(r"C:\Users\Colby\Desktop\Neu-work\AE_notes\plots\AvgLoss vs Latent (11 vars).png")
        plt.close()
    else:
        plt.show()
