from pandas import options, DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from statistics import mean
from commonFunctions import returnTestSamplesSplitIntoSignalAndBackground, compareManyHistograms, returnBestCutValue

def process(dfhiggs, dfqcd, vars):
#Normalize dataframes seperately
    preprocessing.MinMaxScaler().fit_transform(dfhiggs[vars])
    dihiggs_set = DataFrame(dfhiggs[vars], columns=dfhiggs[vars].columns)
    preprocessing.MinMaxScaler().fit_transform(dfqcd[vars])
    qcd_set = DataFrame(dfqcd[vars], columns=dfqcd[vars].columns)
    return qcd_set, dihiggs_set

def epoch_history(history):
#Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('ff-NN Accuracy')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
#Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('ff-NN Model Loss')
    plt.ylabel('Loss [A.U.]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

def significance(qcd_set, dihiggs_set, testing_fraction, vars, model):
    # Sort top 10 variables
    higgstopvars = dihiggs_set[vars]
    qcdtopvars = qcd_set[vars]
    # Insert binary column into each df
    options.mode.chained_assignment = None  # Removes SettingWithCopyWarning which is a false positive
    higgstopvars['Result'] = 1
    qcdtopvars['Result'] = 0
    # Append qcd df to higgs df
    dataset = higgstopvars.append(qcdtopvars)
    # Normalize data
    dataset[vars] = preprocessing.MinMaxScaler().fit_transform(dataset[vars])
    normal_set_df = DataFrame(dataset, columns=dataset.columns)
    # Seperate dataset from results
    data = normal_set_df.loc[:, normal_set_df.columns != 'Result']
    label = normal_set_df.loc[:, 'Result']
    # Split data into training, validation, and testing sets
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=testing_fraction,
                                                                      shuffle=True)
    hh_data_test, hh_labels_test, qcd_data_test, qcd_labels_test = \
        returnTestSamplesSplitIntoSignalAndBackground(data_test, label_test)
    pred_hh = model.predict(hh_data_test)
    pred_qcd = model.predict(qcd_data_test)
    # Plot significance histograms
    _nBins = 40
    predictionResults = {'hh_pred': pred_hh, 'qcd_pred': pred_qcd}
    compareManyHistograms(predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'ff-NN Score', 0, 1,
                          _nBins, _yMax=5, _normed=True, _savePlot=False)
    # Show significance
    returnBestCutValue('ff-NN', pred_hh.copy(), pred_qcd.copy(), _minBackground=400e3,
                       _testingFraction=testing_fraction)