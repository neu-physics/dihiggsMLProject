##  Author:  Ben Tannenwald
##  Date:    Nov 8 2019
##  Purpose: Class to hold common functions for dihiggs work, e.g. lumi-scaling

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def getLumiScaleFactor( _testingFraction=1., _isDihiggs=True ):
    """ function to return lumi-scale for events used in testing and significance calculations """

    # *** 0. Set number of events and total HL-LHC lumi
    lumi_HLLHC = 3000 #fb-1
    nEventsGen = 500e3

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

    # *** 1. Take first nEventsForXGB events for passing to XGB 
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

    # *** 1. Make equal-sized samples
    nTotalEvents = min( len(signal_reduced), len(bkg_reduced))
    print(nTotalEvents, len(signal_reduced), len(bkg_reduced))
    
    signal_reducedForSplit  = signal_reduced[:nTotalEvents]
    bkg_reducedForSplit     = bkg_reduced[:nTotalEvents]
    signal_labelsForSplit   = signal_labels[:nTotalEvents]
    bkg_labelsForSplit       = bkg_labels[:nTotalEvents]

    # *** 2. Combine bkg+signal for passing to XGB 
    all_dataForSplit  = signal_reducedForSplit.append(bkg_reducedForSplit)
    all_labelsForSplit   = signal_labelsForSplit.append(bkg_labelsForSplit)


    # *** 3. Make test/train split
    data_train, data_test, labels_train, labels_test = train_test_split(all_dataForSplit, all_labelsForSplit, test_size=_fractionEventsForTesting, shuffle= True)
    
    # *** 3. Sanity check
    print(len(all_dataForSplit), 'rows of total data with ', len(all_labelsForSplit), 'labels [Train+Test]')
    print(len(data_train), 'rows of training data with ', len(labels_train), 'labels [Train]')
    print(len(data_test), 'rows of testing data with ', len(labels_test), 'labels [Test]')
    
    
    return data_train, data_test, labels_train, labels_test


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
    plt.ylabel('N_events')
    _bins = np.linspace(_xMin, _xMax, _nBins)
   
    y_max = 0
    for iLabel in _labels:
        plt.hist(_dict[iLabel], _bins, alpha=0.5, density=_normed, label= iLabel+' Events')
        
        # get values of histgoram to find greatest y
        #_y, _x, _ = plt.hist(_dict[iLabel])
        #if (_y.max() > y_max):
        #    y_max = _y.max()
    
    # set max y-value of histogram so there's room for legend
    axes = plt.gca()
    axes.set_ylim([0,_yMax])
    #plt.ylim([0,1.2*y_max])
    
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

    _nSignal = sum( value > _bestCutValue for value in _signal) * _signalLumiscale
    _nBackground = sum( value > _bestCutValue for value in _background) * _bkgLumiscale
    #print(_nSignal, _nBackground, _nSignal/np.sqrt(_nBackground), _bestCutValue)
    print('nSig = {0} , nBkg = {1} with significance = {2} for {3} score > {4}'.format(_nSignal, _nBackground, _nSignal/np.sqrt(_nBackground), _variable, _bestCutValue) )
          
    return _bestSignificance, _bestCutValue
