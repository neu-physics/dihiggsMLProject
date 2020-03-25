##  Author:  Ben Tannenwald
##  Date:    Nov 8 2019
##  Purpose: Class to hold common functions for dihiggs work, e.g. lumi-scaling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def getLumiScaleFactor( _testingFraction=1., _isDihiggs=True ):
    """ function to return lumi-scale for events used in testing and significance calculations """

    # *** 0. Set number of events and total HL-LHC lumi
    lumi_HLLHC = 3000 #fb-1
    hh_nEventsGen = 500e3
    qcd_nEventsGen = 2e6
    #qcd_nEventsGen = 500e3

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
    data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=_fractionEventsForTesting, shuffle=True, random_state=30)

    # *** 3. Sanity check
    print(len(all_dataForSplit), 'rows of total data with ', len(all_labelsForSplit), 'labels [Train+Test]')
    print(len(data_train), 'rows of training data with ', len(labels_train), 'labels [Train]')
    print(len(data_test), 'rows of testing data with ', len(labels_test), 'labels [Test]')
    
    
    return data_train, data_test, data_val, labels_train, labels_test, labels_val


def compareManyHistograms( _dict, _labels, _nPlot, _title, _xtitle, _xMin, _xMax, _nBins, _yMax = 4000, _normed=False, _savePlot=False):
       
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
    
    # store figure copy for later saving
    fig = plt.gcf()
    
    # draw interactively
    plt.show()
    
    #save an image file
    if(_savePlot):
        _scope    = _title.split(' ')[0].lower()
        _variable = _xtitle.lstrip('Jet Pair').replace(' ','').replace('[GeV]','').replace('(','_').replace(')','')
        _filename  = _scope + '_' + _variable
        if _normed:
            _filename = _filename + '_norm'
        fig.savefig( _filename+'.png', bbox_inches='tight' )
    
    
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
        _nSignal = sum( value > iCutValue for value in _signal) * _signalLumiscale
        _nBackground = sum( value > iCutValue for value in _background) * _bkgLumiscale
        
        # safety check to avoid division by 0
        if _nBackground < _minBackground: # 500 is semi-random choice.. it's where one series started to oscillate
            #print("continued on {0}".format(iCutValue))
            continue
        
        #if _method == 'S/sqrt(B)':
        #    print(_nSignal, _nBackground, iCutValue, (_nSignal / np.sqrt(_nBackground)), (_nSignal / np.sqrt(_nSignal + _nBackground)))
        
        if _method == 'S/B' and (_nSignal / _nBackground) > _bestSignificance:
            _bestSignificance = (_nSignal / _nBackground)
            _bestCutValue = iCutValue
        elif _method == 'S/sqrt(B)' and (_nSignal / np.sqrt(_nBackground)) > _bestSignificance:
            _bestSignificance = (_nSignal / np.sqrt(_nBackground))
            _bestCutValue = iCutValue
        elif _method == 'S/sqrt(S+B)' and (_nSignal / np.sqrt(_nSignal + _nBackground)) > _bestSignificance:
            _bestSignificance = (_nSignal / np.sqrt(_nSignal + _nBackground))
            _bestCutValue = iCutValue
                
        #print(iCutValue, _nSignal, _nBackground, (_nSignal / np.sqrt(_nBackground)))

    # ** Raw numbers
    _nSignal_raw = sum( value > _bestCutValue for value in _signal) 
    _nBackground_raw = sum( value > _bestCutValue for value in _background) 
    # ** lumi-scaled numbers
    _nSignal = _nSignal_raw * _signalLumiscale
    _nBackground = _nBackground_raw * _bkgLumiscale

    _significance = _nSignal/np.sqrt(_nBackground)
    _sigError = _significance * np.sqrt( 1/_nSignal_raw + 1/(4*_nBackground_raw) )

    #print(_nSignal, _nBackground, _nSignal/np.sqrt(_nBackground), _bestCutValue)

    
    print('nSig = {0} , nBkg = {1} with significance = {2} +/- {3} for {4} score > {5}'.format(_nSignal, _nBackground, _significance, _sigError, _variable, _bestCutValue) )
          
    return _bestSignificance, _bestCutValue


def importDatasets( _hhLabel = '500k', _qcdLabel = '2M', _pileup='0PU'):
    """ function to import datasets from .csv files"""

    #_qcd_csv_files = ['/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_1of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_1of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_2of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_2of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_3of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_3of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_4of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_4of5.csv',
    #                 '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_5of5/qcd_outputDataForLearning_ppTo4b_CMSPhaseII_0PU_top4Tags_store8jets_5of5.csv'
    #]
    #_qcd_raw = pd.concat(map(pd.read_csv, _qcd_csv_files))

    # reconstruction opts: >= 4 tags, store 10 jets, use top4 tagged then highest in pt
    qcd_string = '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_2MEvents_0PU_v2-05__top4inPt-4tags-10jets_combined_csv.csv' if _pileup=='0PU' else '/home/btannenw/Desktop/ML/dihiggsMLProject/data/ppTo4b_2MEvents_200PU_v2-05__top4inPt-4tags-10jets_combined_csv.csv'
    hh_string = '/home/btannenw/Desktop/ML/dihiggsMLProject/data/pp2hh4b_500kEvents_0PU_v2-05__top4inPt-4tags-10jets_combined_csv.csv' if _pileup=='0PU' else '/home/btannenw/Desktop/ML/dihiggsMLProject/data/pp2hh4b_500kEvents_200PU_v2-05__top4inPt-4tags-10jets_combined_csv.csv'

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
