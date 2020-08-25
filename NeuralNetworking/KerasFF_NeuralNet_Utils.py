from pandas import options, DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from statistics import mean
from commonFunctions import returnTestSamplesSplitIntoSignalAndBackground, compareManyHistograms, returnBestCutValue
from zipfile import ZipFile
import os, sys

def extractfiles(dihiggs_filepath):
#Extract dihiggs csv file from zip
    with ZipFile(dihiggs_filepath, 'r') as zip:
        zip.extractall()
        dihiggs_file = zip.namelist()[0]

    return dihiggs_file


def deletefiles(dihiggs_file):
#Deletes files from current directory
    if os.path.exists(dihiggs_file):
        os.remove(dihiggs_file)
    else:
        print("Cannot delete file, it does not exist")


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


def process(dfhiggs, dfqcd, vars, testing_fraction, nJetsFunction=False, nJetstoConsider=False, nBTagsFunction=False,
            nBTagstoConsider=False, equal_qcd_dihiggs_samples=False):
    print("Dataframe shapes before preprocessing:\nDihiggs = " + str(dfhiggs.shape) + "\nQCD = " + str(dfqcd.shape) +
          "\n==============================\n")
    # If removeNJets is False, does not change the csv, otherwise it removes all jets > 'removeNJets'
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

# Sort top 10 variables
    higgstopvars = higgs_set[vars]
    qcdtopvars = qcd_set[vars]

    # If equal_qcd_dihiggs_samples is False, does not change the csv, otherwise qcd and dihiggs samples will be equal
    if equal_qcd_dihiggs_samples == False:
        pass
    elif equal_qcd_dihiggs_samples == True:
        qcdlen = len(qcdtopvars)
        higgslen = len(higgstopvars)
        if qcdlen > higgslen:
            qcdtopvars = qcdtopvars.sample(n=higgslen)
            print("QCD samples reduced to equal dihiggs samples. QCD length = " + str(
                len(qcdtopvars)) + ", dihiggs length = " + str(len(higgstopvars)))
        else:
            higgstopvars = higgstopvars.sample(n=qcdlen)
            print("Dihiggs samples reduced to equal QCD samples. QCD length = " + str(
                len(qcdtopvars)) + ", dihiggs length = " + str(len(higgstopvars)))
    else:
        print("Error, 'equal_qcd_dihiggs_samples' is a boolean statement, parameter must be either true or false.")
        sys.exit()

# Insert binary column into each df
    options.mode.chained_assignment = None  # Removes SettingWithCopyWarning which is a false positive
    higgstopvars['Result'] = 1
    qcdtopvars['Result'] = 0
# Append qcd df to higgs df
    dataset = higgstopvars.append(qcdtopvars)
# Normalize data
    dataset[vars] = preprocessing.scale(dataset[vars])
    normal_set_df = DataFrame(dataset, columns=dataset.columns)
# Seperate dataset from results
    data = normal_set_df.loc[:, normal_set_df.columns != 'Result']
    label = normal_set_df.loc[:, 'Result']
# Split data into training, validation, and testing sets
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=testing_fraction, shuffle=True)  # 80% train, 20% test
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=testing_fraction, shuffle=True)  # 80% train, 20% validation

    return data_test, data_train, data_val, label_test, label_train, label_val

def significance(model, data_test, label_test, testing_fraction):
    hh_data_test, hh_labels_test, qcd_data_test, qcd_labels_test = \
        returnTestSamplesSplitIntoSignalAndBackground(data_test, label_test)
    pred_hh = model.predict(hh_data_test)
    pred_qcd = model.predict(qcd_data_test)
    # Plot significance histograms
    _nBins = 40
    predictionResults = {'hh_pred': pred_hh, 'qcd_pred': pred_qcd}
    compareManyHistograms(predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'ff-NN Score', 0, 1,
                          _nBins, _yMax=5, _normed=True)
    # Show significance
    returnBestCutValue('ff-NN', pred_hh.copy(), pred_qcd.copy(), _minBackground=400e3,
                       _testingFraction=testing_fraction)

def accuracy(model, data_train, label_train, data_test, label_test):
    trainscores_raw = []
    testscores_raw = []
    x = 0
    while x <= 4:
        trainscores_raw.append(model.evaluate(data_train, label_train)[1]*100)
        testscores_raw.append(model.evaluate(data_test, label_test)[1]*100)
        x+=1
    trainscores = ("Training Accuracy: %.2f%%\n" % (mean(trainscores_raw)))
    testscores = ("Testing Accuracy: %.2f%%\n" % (mean(testscores_raw)))

    return trainscores, testscores

def epoch_history(history):
#Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('ff-NN Accuracy')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
#Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('ff-NN Model Loss')
    plt.ylabel('Loss [A.U.]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def roc_plot(model, data_test, label_test): #Plot roc curve with auc score
    predictions = model.predict(data_test)
    fpr, tpr, threshold = roc_curve(label_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
