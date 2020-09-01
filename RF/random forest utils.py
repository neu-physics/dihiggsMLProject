from sklearn.preprocessing import scale
from sklearn.model_selection import learning_curve
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import commonFunctions
from zipfile import ZipFile
import os
import sys

seed = 7
np.random.seed(seed)

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

    # If equal_qcd_dihiggs_samples is False, does not change the csv, otherwise qcd and dihiggs samples will be equal
    if equal_qcd_dihiggs_samples == False:
        pass
    elif equal_qcd_dihiggs_samples == True:
        qcdlen = len(qcd_set)
        higgslen = len(higgs_set)
        if qcdlen > higgslen:
            qcd_set = qcd_set.sample(n=higgslen)
            print("QCD samples reduced to equal dihiggs samples. QCD length = " + str(
                len(qcd_set)) + ", dihiggs length = " + str(len(higgs_set)))
        else:
            higgs_set = higgs_set.sample(n=qcdlen)
            print("Dihiggs samples reduced to equal QCD samples. QCD length = " + str(
                len(qcd_set)) + ", dihiggs length = " + str(len(higgs_set)))
    else:
        print("Error, 'equal_qcd_dihiggs_samples' is a boolean statement, parameter must be either true or false.")
        sys.exit()

    higgs_set['isSignal'] = 1
    qcd_set['isSignal'] = 0
    data_train, data_test, label_train, label_test = \
        commonFunctions.makeTestTrainSamplesWithUserVariables(higgs_set, qcd_set, vars, testing_fraction)
    data_train_norm = scale(data_train.values)
    data_test_norm = scale(data_test.values)
    label_train = label_train.to_numpy()
    label_test = label_test.to_numpy()

    return data_train_norm, data_test_norm, label_train, label_test


def trainBDT(X, y, X_val, y_val, param, min_background):
    #Train trees
    evallist = [(X, y),(X_val, y_val)]
    model = xgb.XGBRFClassifier(**param)
    model.fit(X, y.ravel(), eval_set=evallist, verbose=True)

    #Get significance data
    ypred = model.predict(X_val)
    predictions = [round(value) for value in ypred]
    accuracy = accuracy_score(y_val, predictions)
    print("The training accuaracy is: {}".format(accuracy))
    conf_matrix = confusion_matrix(y_val, predictions)
    print("The confusion matrix: {}".format(conf_matrix))
    print("The precision is: {}".format(precision_score(y_val, predictions)))
    plot_BDTScore(X_val.copy(), y_val.copy(), model, min_background)

    return model, predictions


def plot_BDTScore(X_val, y_val, model, min_background):
    sig_index = np.asarray(np.where(y_val == 1))[0, :]
    bkg_index = np.asarray(np.where(y_val == 0))[0, :]
    X_sig = X_val[sig_index, :]
    X_bkg = X_val[bkg_index, :]
    pred_sig = model.predict_proba(X_sig)[:, 1]
    pred_bkg = model.predict_proba(X_bkg)[:, 1]
    commonFunctions.returnBestCutValue('RF', pred_sig.copy(), pred_bkg.copy(), _testingFraction=0.3, _minBackground=min_background)
    plt.hist(pred_sig, bins=100, alpha=0.5, density=True, label="signal")
    plt.hist(pred_bkg, bins=100, alpha=0.5, density=True, label="background")
    plt.legend(loc="best")
    plt.title("RF score")
    plt.show()


def plot_feature_importances(model, vars):
    #Set feature names
    model.get_booster().feature_names = vars

    #Plot feature weights (How often the feature was used to make decisions)
    xgb.plot_importance(model.get_booster(), title='Weight')
    plt.show()

    #Plot feature gain (Importance of the variable when generating a prediction)
    xgb.plot_importance(model.get_booster(), title='Gain', importance_type='gain')
    plt.show()


def plot_learning_curve(X, y, param, nClus):
    train_sizes, train_scores, test_scores = learning_curve(xgb.XGBClassifier(**param), X, y, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing score")
    plt.legend(loc="best")
    plt.title("{} clusters learning curve".format(nClus))
    plt.show()


def plotCostFunc_kClus(data, nClus):
    # find out the "best" value of n_clusters to perform k-means clustering
    cost = []
    for i in range(1, nClus + 1):
        ki = KMeans(n_clusters=i, random_state=0).fit(data)
        cost.append(ki.inertia_)
    plt.plot(range(1, nClus + 1), cost, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('cost function')
    plt.show()


def KClustering(X, y, X_test, y_test, nClusters, usePCA, n_vars):
    if (usePCA):
        # process data with PCA
        # find the number of features that keep 95% variance
        print("Doing PCA...")
        variance_threshold = 0.85
        num_components = n_vars
        pca_trail = PCA()
        pca_trail.fit(X)
        var = np.cumsum(pca_trail.explained_variance_ratio_)
        for n_com in range(1, len(var) - 1):
            if (var[n_com] > variance_threshold):
                num_components = n_com
                break

        print("Doing k-means clustering with {0} features...".format(num_components))
        pca = PCA(n_components=num_components)
        pca.fit(X)
        X_train_pca = pca.transform(X)
        X_test_pca = pca.transform(X_test)
        print("Shape of new training dataset: {}".format(X_train_pca.shape))
        print("Shape of new testing dataset: {}".format(X_test_pca.shape))
        # do the k-means clustering
        kmeans = KMeans(n_clusters=nClusters, random_state=0, verbose=0).fit(X_train_pca)
        score_train = kmeans.transform(X_train_pca)
        score_test = kmeans.transform(X_test_pca)
    else:
        # do k-means clustering
        print("Doing k-means clustering...")
        kmeans = KMeans(n_clusters=nClusters, random_state=0, verbose=0).fit(X)
        score_train = kmeans.transform(X)
        score_test = kmeans.transform(X_test)

    score_train_norm = scale(score_train)
    score_test_norm = scale(score_test)
    y_np = y.to_numpy()
    y_test_np = y_test.to_numpy()
    print("Finished clustering. :)")

    return score_train_norm, y_np, score_test_norm, y_test_np

def roc_plot(label_test, predictions): #Plot roc curve with auc score
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
