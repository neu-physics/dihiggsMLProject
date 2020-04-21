##  Author:  Ben Tannenwald
##  Date:    Nov 8 2019
##  Purpose: Class to hold common functions for dihiggs work, e.g. lumi-scaling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


def getLumiScaleFactor( _testingFraction=1., _isDihiggs=True, _nEventsGen=-1 ):
    """ function to return lumi-scale for events used in testing and significance calculations """

    # *** 0. Set number of events and total HL-LHC lumi
    lumi_HLLHC = 3000 #fb-1

    hh_nEventsGen = 500e3
    qcd_nEventsGen = 2e6

    if _isDihiggs and _nEventsGen > 0:
        hh_nEvetsGen = _nEventsGen
    if _isDihiggs==False and _nEventsGen > 0:
        qcd_nEvetsGen = _nEventsGen

    nEventsGen = hh_nEventsGen if _isDihiggs else qcd_nEventsGen
    
    # *** 1. Set appropriate cross-section for sample
    hh_xsec = 12.36 # fb
    qcd_xsec = 441866.0 # fb
    xsec = hh_xsec if _isDihiggs else qcd_xsec

    # *** 2. Caclulate sample lumi and nominal lumi-scale
    lumi_sample = nEventsGen / xsec
    lumiscale   = lumi_HLLHC / lumi_sample

    # *** 3. Scale if using subset of selected events for testing
    lumiscale = lumiscale / _testingFraction


    #lumiscale = lumi_HLLHC if not _isDihiggs else lumi_HLLHC*4.116970394139372e-05
    return lumiscale



def makeEqualSamplesWithUserVariables(signal_raw, bkg_raw, userVariables, nEventsForXGB):
    """function to return 4 dataframes containing user-specified variables and number of events: 1 signal for training, 1 bkg for training, 1 signal for plotting, 1 bkg for plotting"""
    
    # *** 0. Reduce dataframes to only desired variables
    signal_reduced  = signal_raw[userVariables]
    bkg_reduced     = bkg_raw[userVariables]
    signal_labels   = signal_raw[ ['isSignal'] ]
    bkg_labels      = bkg_raw[ ['isSignal'] ]

    # *** 1A. Take first nEventsForXGB events for passing 1:1 signal-to-background to XGB 
    signal_reducedForXGB  = signal_reduced[:nEventsForXGB]
    bkg_reducedForXGB     = bkg_reduced[:nEventsForXGB]
    signal_labelsForXGB   = signal_labels[:nEventsForXGB]
    bkg_labelsForXGB      = bkg_labels[:nEventsForXGB]


    # *** 2. Combine bkg+signal for passing to XGB 
    all_reducedForXGB  = signal_reducedForXGB.append(bkg_reducedForXGB)
    all_labelsForXGB   = signal_labelsForXGB.append(bkg_labelsForXGB)

    
    # ** 3. Use additional events for unambiguous testing 
    signal_reducedForPlots  = signal_reduced[nEventsForXGB:len(bkg_reduced)]
    bkg_reducedForPlots     = bkg_reduced[nEventsForXGB:len(bkg_reduced)]
    signal_labelsForPlots   = signal_labels[nEventsForXGB:len(bkg_reduced)]
    bkg_labelsForPlots      = bkg_labels[nEventsForXGB:len(bkg_reduced)]

    # *** 4. Sanity check
    print(len(all_reducedForXGB), 'rows of data with ', len(all_labelsForXGB), 'labels [XGB]')
    print(len(signal_reducedForPlots), 'rows of signal data with ', len(bkg_labelsForPlots), 'rows of background [Plots]')

    
    return all_reducedForXGB, all_labelsForXGB, signal_reducedForPlots, signal_labelsForPlots, bkg_reducedForPlots, bkg_labelsForPlots


def makeTestTrainSamplesWithUserVariables(signal_raw, bkg_raw, userVariables, _fractionEventsForTesting):
    """function to return 4 dataframes containing user-specified variables and number of events: 1 mixed signal+background for training, 1 mixed signal+background for testing"""
    
    # *** 0. Reduce dataframes to only desired variables
    signal_reduced  = signal_raw[userVariables]
    bkg_reduced     = bkg_raw[userVariables]
    signal_labels   = signal_raw[ ['isSignal'] ]
    bkg_labels      = bkg_raw[ ['isSignal'] ]

    ## *** 1A. Make equal-sized samples
    #nTotalEvents = min( len(signal_reduced), len(bkg_reduced))
    #print("N_sig = {0} , N_bkg = {1}".format(len(signal_reduced), len(bkg_reduced)))
    
    #signal_reducedForSplit  = signal_reduced[:nTotalEvents]
    #bkg_reducedForSplit     = bkg_reduced[:nTotalEvents]
    #signal_labelsForSplit   = signal_labels[:nTotalEvents]
    #bkg_labelsForSplit       = bkg_labels[:nTotalEvents]

    # *** 1B. Take all 
    print("N_sig = {0} , N_bkg = {1}".format(len(signal_reduced), len(bkg_reduced)))
    signal_reducedForSplit  = signal_reduced
    bkg_reducedForSplit     = bkg_reduced
    signal_labelsForSplit   = signal_labels
    bkg_labelsForSplit      = bkg_labels

    # *** 2. Combine bkg+signal for passing to Split 
    all_dataForSplit  = signal_reducedForSplit.append(bkg_reducedForSplit)
    all_labelsForSplit   = signal_labelsForSplit.append(bkg_labelsForSplit)

    # *** 3. Make test/train split
    data_train, data_test, labels_train, labels_test = train_test_split(all_dataForSplit, all_labelsForSplit, test_size=_fractionEventsForTesting, shuffle= True, random_state=30)
    
    # *** 3. Sanity check
    print(len(all_dataForSplit), 'rows of total data with ', len(all_labelsForSplit), 'labels [Train+Test]')
    print(len(data_train), 'rows of training data with ', len(labels_train), 'labels [Train]')
    print(len(data_test), 'rows of testing data with ', len(labels_test), 'labels [Test]')
    
    
    return data_train, data_test, labels_train, labels_test


def compareManyHistograms( _dict, _labels, _nPlot, _title, _xtitle, _xMin, _xMax, _nBins, _yMax = 4000, _normed=False, savePlot=False, saveDir='', writeSignificance=False, _testingFraction=1.0):
       
    if len(_dict.keys()) < len(_labels):
        print ("!!! Unequal number of arrays and labels. Learn to count better.")
        return 0
    
    plt.figure(_nPlot)
    if _normed:
        plt.title(_title + ' (Normalized)')
    else:
        plt.title(_title)
    plt.xlabel(_xtitle)
    plt.ylabel('Events/Bin [A.U.]')
    _bins = np.linspace(_xMin, _xMax, _nBins)

    
    for iLabel in _labels:
        if _normed:
            _weights = np.ones_like(_dict[iLabel]) / len(_dict[iLabel])
            _counts_final, _bins_final, _patches_final = plt.hist(_dict[iLabel], bins=_bins, weights=_weights, alpha=0.5, label= iLabel+' Events')
            #print(sum(_dict[iLabel]*_weights), sum(_counts_final))
            
        else:
            plt.hist(_dict[iLabel], bins=_bins, alpha=0.5, label= iLabel+' Events')
                    

    # set max y-value of histogram so there's room for legend
    _yMax = 0.15 if _normed else _yMax
    axes = plt.gca()
    axes.set_ylim([0,_yMax])
        
    #draw legend
    plt.legend(loc='upper left')
    #plt.text(.1, .1, s1)

    # ** X. Add significance and cut if requested
    if writeSignificance==True:
        _pred_sig = _dict['hh_pred']
        _pred_bkg = _dict['qcd_pred']

        sig, cut, err = returnBestCutValue(_xtitle, _pred_sig.copy(), _pred_bkg.copy(), _minBackground=200, _testingFraction=_testingFraction)
        plt.text(x=0.6, y=0.12, s= '$\sigma$ = {} $\pm$ {}\n (score > {})'.format(round(sig, 2), round(err, 2), round(cut, 3)), fontsize=13 )
            
    # store figure copy for later saving
    fig = plt.gcf()
    
    # draw interactively
    plt.show()
    
    #save an image file
    if(savePlot):
        _scope    = _title.split(' ')[0].lower()
        _variable = _xtitle.lstrip('Jet Pair').replace(' ','').replace('[GeV]','').replace('(','_').replace(')','')
        _filename  = _scope + '_' + _variable
        if _normed:
            _filename = _filename + '_norm'
        fig.savefig( saveDir + '/' + _filename+'.png', bbox_inches='tight' )
    
    
    return


def returnBestCutValue( _variable, _signal, _background, _method='S/sqrt(B)', _minBackground=500, _testingFraction=1.):
    """find best cut according to user-specified significance metric"""

    _signalLumiscale = getLumiScaleFactor( _testingFraction, _isDihiggs = True)
    _bkgLumiscale = getLumiScaleFactor( _testingFraction, _isDihiggs = False)
    
    _bestSignificance = -1
    _bestCutValue = -1
    _massWidth = 30 #GeV
    _nTotalSignal =len(_signal) 
    _nTotalBackground =len(_background) 
    _cuts = []
    _sortedSignal = np.sort(_signal )
    _sortedBackground = np.sort(_background )

    print(_nTotalSignal, _nTotalBackground)
    _minVal = min( min(_sortedSignal), min(_sortedBackground) )
    _maxVal = max( max(_sortedSignal), max(_sortedBackground) )
    
    if 'mass' in _variable:
        _stepSize = 0.05 if 'mass' not in _variable else 5
        _cuts = list(range(_minVal, _maxVal, _stepSize))
    else:
        _cuts = np.linspace(_minVal, _maxVal, 100)
    
    #print(_minVal, _maxVal)

    for iCutValue in _cuts:
        _nSignal = float(sum( value > iCutValue for value in _signal) * _signalLumiscale)
        _nBackground = float(sum( value > iCutValue for value in _background) * _bkgLumiscale)
        
        # safety check to avoid division by 0
        if _nBackground < _minBackground: # 500 is semi-random choice.. it's where one series started to oscillate
            #print("continued on {0}".format(iCutValue))
            continue
        
        #if _method == 'S/sqrt(B)':
        #    print(_nSignal, _nBackground, iCutValue, (_nSignal / np.sqrt(_nBackground)), (_nSignal / np.sqrt(_nSignal + _nBackground)))
        
        if _method == 'S/B' and (_nSignal / _nBackground) > _bestSignificance:
            _bestSignificance = float(_nSignal / _nBackground)
            _bestCutValue = float(iCutValue)
        elif _method == 'S/sqrt(B)' and (_nSignal / np.sqrt(_nBackground)) > _bestSignificance:
            _bestSignificance = float(_nSignal / np.sqrt(_nBackground))
            _bestCutValue = float(iCutValue)
        elif _method == 'S/sqrt(S+B)' and (_nSignal / np.sqrt(_nSignal + _nBackground)) > _bestSignificance:
            _bestSignificance = float(_nSignal / np.sqrt(_nSignal + _nBackground))
            _bestCutValue = float(iCutValue)
                
        #print(iCutValue, _nSignal, _nBackground, (_nSignal / np.sqrt(_nBackground)))

    # ** Raw numbers
    _nSignal_raw = sum( value > _bestCutValue for value in _signal) 
    _nBackground_raw = sum( value > _bestCutValue for value in _background) 
    # ** lumi-scaled numbers
    _nSignal = float(_nSignal_raw * _signalLumiscale)
    _nBackground = float(_nBackground_raw * _bkgLumiscale)

    _significance = float(_nSignal/np.sqrt(_nBackground))
    _sigError = float(_significance * np.sqrt( 1/_nSignal_raw + 1/(4*_nBackground_raw) ))

    #print(_nSignal, _nBackground, _nSignal/np.sqrt(_nBackground), _bestCutValue)

    
    print('nSig = {0} , nBkg = {1} with significance = {2} +/- {3} for {4} score > {5}'.format( round(_nSignal, 2), round(_nBackground, 2), round(_significance, 3), round(_sigError, 3), _variable, round(float(_bestCutValue), 3)) )
          
    return _bestSignificance, float(_bestCutValue), _sigError


def importDatasets( _hhLabel = '500k', _qcdLabel = '2M', _pileup='0PU', _btags='4'):
    """ function to import datasets from .csv files"""

    #_qcd_csv_files = ['/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_1of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_1of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_2of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_2of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_3of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_3of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_4of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_4of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_5of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_5of5.csv'
    #]
    #_qcd_raw = pd.concat(map(pd.read_csv, _qcd_csv_files))

    # reconstruction opts: >= 4 tags, store 10 jets, use top4 tagged then highest in pt
    qcd_string = '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_2MEvents_0PU_v2-05__top4inPt-4tags-10jets_combined_csv.csv' if _pileup=='0PU' else '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_2MEvents_200PU_v2-05__top4inPt-'+_btags+'tags-10jets_combined_csv.csv'
    hh_string = '/home/btannenw/Desktop/ML/dihiggsMLProject/data/pp2hh4b_500kEvents_0PU_v2-05__top4inPt-4tags-10jets_combined_csv.csv' if _pileup=='0PU' else '/home/btannenw/Desktop/ML/dihiggsMLProject/data/pp2hh4b_500kEvents_200PU_v2-05__top4inPt-'+_btags+'tags-10jets_combined_csv.csv'

    print("Dihiggs file: ", hh_string)
    print("QCD file: ", qcd_string)
    
    _qcd_raw = pd.read_csv( qcd_string )
    _qcd_raw['isSignal'] = 0
    _qcd_raw = _qcd_raw[_qcd_raw.columns.drop(list(_qcd_raw.filter(regex='gen')))] # drop truth quark info
    
    _hh_raw = pd.read_csv( hh_string )
    _hh_raw['isSignal'] = 1
    _hh_raw = _hh_raw.drop('isMatchable', 1)
    _hh_raw = _hh_raw[_hh_raw.columns.drop(list(_hh_raw.filter(regex='gen')))] # drop truth quark info


    return _hh_raw, _qcd_raw

def returnTestSamplesSplitIntoSignalAndBackground(_test_data, _test_labels):
    
    _test_data = _test_data.copy()

    if type(_test_data) != np.ndarray: # traditional NN and BDT approachs --> passing pandas df directly to function
        _test_data['isSignal'] = _test_labels
    
        _test_signal_data = _test_data[ _test_data.isSignal==1 ]
        _test_bkg_data    = _test_data[ _test_data.isSignal==0 ]
        
        _test_signal_labels = _test_signal_data.isSignal
        _test_bkg_labels    = _test_bkg_data.isSignal
        
        _test_signal_data = _test_signal_data.drop('isSignal', axis=1)
        _test_bkg_data = _test_bkg_data.drop('isSignal', axis=1)

    elif type(_test_data) == np.ndarray: # LBN Network approach --> passing numpy array

        print(np.shape(_test_labels))
        if np.shape(_test_labels)[1] == 2:
            _test_signal_data   = [ _eventVectors for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[0] == 1]
            _test_bkg_data      = [ _eventVectors for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[1] == 1]
            
            _test_signal_labels = [ _signalEncoding for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[0] == 1]
            _test_bkg_labels    = [ _signalEncoding for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[1] == 1]
        elif np.shape(_test_labels)[1] == 1:
            _test_signal_data   = [ _eventVectors for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[0] == 1]
            _test_bkg_data      = [ _eventVectors for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[0] == 0]

            _test_signal_labels = [ _signalEncoding[0] for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[0] == 1]
            _test_bkg_labels    = [ _signalEncoding[0] for _eventVectors,_signalEncoding in zip(_test_data, _test_labels) if _signalEncoding[0] == 0]

            
    return _test_signal_data.copy(), _test_signal_labels.copy(), _test_bkg_data.copy(), _test_bkg_labels.copy()


def makeHistoryPlots(_history, _curves=['loss'], _modelName='', savePlot=False, saveDir=''):
    """ make history curves for user-specified training parameters"""

    for curve in _curves:
        plt.plot(_history.history[ curve])
        plt.plot(_history.history['val_'+curve])
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        
        if curve == 'loss':            # summarize history for loss
            plt.title('{} Model Loss'.format(_modelName))
            plt.ylabel('Loss [A.U.]')
            plt.ylim([0, 1])
        elif curve == 'auc':            # summarize history for AUC
            plt.title('{} Model AUC'.format(_modelName))
            plt.ylabel('AUC [A.U.]')
            plt.ylim([0.5, 1])
        elif curve == 'categorical_accuracy': # summarize history for accuracy
            plt.title('{} Accuracy'.format(_modelName))
            plt.ylabel('Accuracy [%]')
            plt.ylim([0.5, 1])
        else:            # summarize history for accuracy
            plt.title('{} {}'.format(_modelName, curve))
            plt.ylabel('{} [A.U.]'.format(curve))
            plt.ylim([0.5, 1])

            
        # store figure copy for later saving
        fig = plt.gcf()

        # draw interactively
        plt.show()
    
        #save an image file
        if(savePlot):
            _filename  = '{}_history_{}'.format(_modelName, curve)
            fig.savefig( saveDir + '/' + _filename+'.png', bbox_inches='tight' )

    return

def makeEfficiencyCurves(*data, _modelName='', savePlot=False, saveDir=''):
    """ make curve of signal efficiency vs background rejection given some inputs"""
    
    # basic plot setup
    #plt.plot([0, 1], [1, 0], color="black", linestyle="--")
    plt.plot( [[0, 0], [1, 1]], color="black", linestyle="--")
    plt.title("{} ROC curve".format(_modelName))
    #plt.xlabel("Signal Efficiency")
    #plt.ylabel("Background Rejection")
    plt.xlabel("Background Efficiency")
    plt.ylabel("Signal Efficiency")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.tick_params(axis="both", direction="in")

    # add data
    for d in data:
        auc = roc_auc_score(d["labels"], d["prediction"])
        label = "{} ({:.3f})".format(d.get("label", "ROC"), auc)
        if len(d["prediction"][0]) == 1:
            roc = roc_curve(d["labels"][:], d["prediction"][:])
        else:
            roc = roc_curve(d["labels"][:, 1], d["prediction"][:, 1])
        fpr, tpr, _ = roc
        #plt.plot(tpr, 1 - fpr, label=label, color=d.get("color", "#118730")) # signal eff vs background rejection
        plt.plot(fpr, tpr, label=label, color=d.get("color", "#118730")) # signal eff vs background eff
        

    # legend
    leg = plt.legend(loc="lower left", fontsize="small")
    
    # store figure copy for later saving
    fig = plt.gcf()

    # draw interactively
    plt.show()
    
    #save an image file
    if(savePlot):
        _filename  = '{}_ROC'.format(_modelName)
        fig.savefig( saveDir + '/' + _filename+'.png', bbox_inches='tight' )


    return


def overlayROCCurves(data, savePlot=False, saveDir=''):
    """ overlay multiple ROC curves given some inputs"""

    # basic plot setup
    plt.plot( [[0, 0], [1, 1]], color="black", linestyle="--")
    plt.title("ROC Curves")
    plt.xlabel("Background Efficiency")
    plt.ylabel("Signal Efficiency")
    plt.xlim(1e-4, 1)
    plt.ylim(1e-3, 1)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.tick_params(axis="both", direction="in")

    
    for d in data:    
        auc = roc_auc_score(d['labels'], d['prediction'])
        label = "{} ({:.3f})".format(d.get("label", "ROC"), auc)
        if len(d["prediction"][0]) == 1:
            roc = roc_curve(d["labels"][:], d["prediction"][:])
        else:
            roc = roc_curve(d["labels"][:, 1], d["prediction"][:, 1])
        fpr, tpr, _ = roc
        plt.plot(fpr, tpr, label=label, color=d.get("color", "#118730")) # signal eff vs background eff

    # legend
    leg = plt.legend(loc="lower right", fontsize="small")

    # store figure copy for later saving
    fig = plt.gcf()

    # draw interactively
    plt.show()

    #save an image file
    if(savePlot):
        _filename  = '{}_ROC'.format(_modelName)
        fig.savefig( saveDir + '/' + _filename+'.png', bbox_inches='tight' )


    return 

