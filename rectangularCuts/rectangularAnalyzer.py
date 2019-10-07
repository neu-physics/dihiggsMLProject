##  Author:  Ben Tannenwald
##  Date:    Oct 7, 2019
##  Purpose: Class to hold functions for testing sequential 1-D cuts to distinguish between dihiggs signal and qcd background

import pandas as pd
import numpy as np
import itertools, csv

massWidth = 30 #GeV

class sequentialOneDimAnalyzer:
    
    def __init__ (self, _inputSignalFile, _inputBackgroundFile, _variables, _isTestRun = False, _metrics=['S/B', 'S/sqrt(B)', 'S/sqrt(S+B)']):
        self.inputSignalData   = pd.read_csv(_inputSignalFile)
        self.inputBkgData      = pd.read_csv(_inputBackgroundFile)
        self.variables         = _variables
        self.isTestRun         = _isTestRun
        self.metrics           = _metrics
        
        # Class Defaults
        self.reducedSignalData = self.inputSignalData[self.variables]
        self.reducedBkgData    = self.inputBkgData[self.variables]
        self.orderedCuts_byMetric   = {}
        self.dictOfCutsByEfficiency = {}


        
    ##############################################################
    ##                           MAIN                           ##
    ##############################################################

    def runAnalysis(self, _dumpPlots=False):

        # *** 0. Tell user what variables under consideration and make two simple plots
        print("Analying variables: ", self.reducedSignalData.columns)
        if _dumpPlots:
            _nBins = 100
            self.reducedSignalData.hist(column=self.variables[2], bins=_nBins)
            self.reducedBkgData.hist(column=self.variables[2], bins=_nBins)

        # *** 1. One-by-one analysis of best cut values
        print("*** 1. Best Single Cut variables evaluated with {0}".format(self.metrics))
        #self.printBestCutValues()

        # *** 2. Some sequential dictionary approach now
        print("*** 2. Get Dictionaries of significance-ordered best cuts using {0}".format(self.metrics))
        #self.storeDictOfOrderedCuts()

        # *** 3. Individual cut approach for best cuts
        print("*** 3. Get Dictionaries of significance-ordered best cuts using {0}".format(self.metrics))
        self.storeDictOfCutsByEfficiency()

        # *** 4. Get S and B yields using cuts from const efficiency approach
        efficiencyToCalculate = 0.90
        print("*** 4. Yields using constant {0}% efficiency cuts".format(efficiencyToCalculate*100))
        self.calculateYieldsAfterCuts(efficiencyToCalculate)

        
        return

    
    ##############################################################
    ##                FUNCTIONS TO RUN ANALYSIS                 ##
    ##############################################################
    def calculateYieldsAfterCuts(self, _efficiency):
        """ use constant efficiency cuts from previous step to calculate yields after all cuts"""

        self.returnNumberSignalAndBackgroundAfterCuts( self.reducedSignalData.copy(), self.reducedBkgData.copy(), self.dictOfCutsByEfficiency[ _efficiency ], 'mass')
        self.returnNumberSignalAndBackgroundAfterCuts( self.reducedSignalData.copy(), self.reducedBkgData.copy(), self.dictOfCutsByEfficiency[ _efficiency ], 'elta')
        self.returnNumberSignalAndBackgroundAfterCuts( self.reducedSignalData.copy(), self.reducedBkgData.copy(), self.dictOfCutsByEfficiency[ _efficiency ])

        return
    

    def storeDictOfCutsByEfficiency(self, _efficiencies=[0.80, 0.85, 0.90], _greaterThanLessThan = '>'):
        """store dict of one-dimensional cuts by user-specified efficiency"""

        for _efficiency in _efficiencies:
            self.dictOfCutsByEfficiency[_efficiency] = self.getCutsForSpecifiedEfficiency(self.reducedSignalData.copy(), self.reducedBkgData.copy(), float(_efficiency),  _greaterThanLessThan)

        return

    
    def storeDictOfOrderedCuts(self):
        """ function to store dict (by metric) of odered best cuts"""

        for _metric in self.metrics:
            self.orderedCuts_byMetric[_metric] = self.returnSignificanceOrderedCutDict( _metric, self.variables.copy(), self.reducedSignalData.copy(), self.reducedBkgData.copy())
            print( self.orderedCuts_byMetric[_metric] )

            return
        
    def printBestCutValues(self):
        """ function to print output of best cut functions"""

        for iColumn in range(0, len(self.reducedSignalData.columns) ):
            varName = self.variables[iColumn]
            sortedSignal = np.sort(self.reducedSignalData[varName].values)
            sortedBackground = np.sort(self.reducedBkgData[varName].values)

            for _metric in self.metrics:
                bestCut, significance = self.returnBestCutValue( varName, sortedSignal, sortedBackground, _metric)
                print ( _metric, varName, bestCut, significance )
 
            print("=====================================")

        return


    
    ##############################################################
    ##              FUNCTIONS TO HANDLE ANALYSIS                ##
    ##############################################################
    def returnCutValueByConstantEfficiency(self, _variable, _signal, _background, _eff, _inequality = '>'):
        """return a cut value based on keeping some constant efficiency of signal"""

        _bestCutValue = -1
        _nTotalSignal =len(_signal) 
        _nTotalBackground =len(_background) 
        _cuts = []
        
        _minVal = int(min(min(_background), min(_signal))) if 'mass' not in _variable else int(min(min(_background), min(_signal))) - int(min(min(_background), min(_signal)))%5
        _maxVal = int(max(max(_background), max(_signal))) if 'mass' not in _variable else int(max(max(_background), max(_signal))) - int(max(max(_background), max(_signal)))%5
        if 'mass' in _variable:
            #_cuts = list(range(_minVal, _maxVal, _stepSize))
            _cuts = list(range(0, _maxVal, 5))
            #print(_maxVal, max(_background), max(_signal))
        elif 'deltaR' in _variable:
            #_cuts = np.linspace(_minVal, _maxVal, 100)
            _cuts = np.linspace(0, 5, 101)
            
            
        for iCutValue in _cuts:
            if _inequality == '<':
                _nSignal = sum( value < iCutValue for value in _signal)
                _nBackground = sum( value < iCutValue for value in _background)
            elif _inequality == '>':
                _nSignal = sum( value > iCutValue for value in _signal)
                _nBackground = sum( value > iCutValue for value in _background)
            else:
                print("Unknown inequality operator {0}. EXITING".format(_inequality))
                return _bestCutValue, _nTotalSignal, _nTotalBackground
            
            # safety check to avoid division by 0
            if _nBackground == 0:
                continue
        
            if _inequality == '<':
                if _nSignal / _nTotalSignal >= _eff:
                    _bestCutValue = iCutValue
                    return _bestCutValue, _nSignal, _nBackground
            elif _inequality== '>':
                if _nSignal / _nTotalSignal < _eff: # passed threshold so return previous cut
                    _nSignalBest = sum( value > _bestCutValue for value in _signal)
                    _nBackgroundBest = sum( value > _bestCutValue for value in _background)
                    return _bestCutValue, _nSignalBest, _nBackgroundBest
                else:
                    _bestCutValue = iCutValue
            
        return _bestCutValue, -1, -1
    
    
    def returnBestCutValue(self, _variable, _signal, _background, _method='S/B'):
        """find best cut according to user-specified significance metric"""
    
        _bestSignificance = -1
        _bestCutValue = -1
        _massWidth = massWidth #GeV
        _nTotalSignal =len(_signal) 
        _nTotalBackground =len(_background) 
        _cuts = []
        _sortedSignal = _signal
        _sortedBackground = _background
        
        _minVal = int(min(min(_sortedBackground), min(_sortedSignal))) if 'mass' not in _variable else int(min(min(_sortedBackground), min(_sortedSignal))) - int(min(min(_sortedBackground), min(_sortedSignal)))%5
        _maxVal = int(max(max(_sortedBackground), max(_sortedSignal))) if 'mass' not in _variable else int(max(max(_sortedBackground), max(_sortedSignal))) - int(max(max(_sortedBackground), max(_sortedSignal)))%5
        if 'mass' in _variable:
            _stepSize = 0.05 if 'mass' not in _variable else 5
            _cuts = list(range(_minVal, _maxVal, _stepSize))
        else:
            _cuts = np.linspace(_minVal, _maxVal, 100)
    
        for iCutValue in _cuts:
            if 'mass' in _variable:
                _nSignal = sum( (value > iCutValue and value < (iCutValue+_massWidth)) for value in _signal) 
                _nBackground = sum( (value > iCutValue and value < (iCutValue+_massWidth)) for value in _background)
                #_nSignal = sum( (value > iCutValue ) for value in _signal)
                #_nBackground = sum( (value > iCutValue) for value in _background)
            else:
                _nSignal = sum( value < iCutValue for value in _signal)
                _nBackground = sum( value < iCutValue for value in _background)

            # temporary fix since samples with different number of events
            #_nSignal = _nSignal / _nTotalSignal
            #_nBackground = _nBackground / _nTotalBackground
            _nSignal = _nSignal * (_nTotalBackground / _nTotalSignal )
            #_nBackground = _nBackground / _nTotalBackground
        
            # safety check to avoid division by 0
            if _nBackground == 0:
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
                
        return _bestSignificance, _bestCutValue

    
    def returnSignificanceOrderedCutDict(self, _method, _varNames, _signalDataFrame, _backgroundDataFrame):
        """function to return list of cuts ordered by descending significance"""
        
        _orderedVariableAndCutDict = {}
        _unprocessedVariables = _varNames
        _signalAfterCuts = _signalDataFrame
        _backgroundAfterCuts = _backgroundDataFrame
        
        while len(_unprocessedVariables)>0:
            _iBestCut = -1
            _iBestSignificance = -1
            _iBestVariable = ''
            print('iteration {0}, signal has {1} rows'.format(len(_unprocessedVariables), len(_signalAfterCuts)))
            print('iteration {0}, background has {1} rows'.format(len(_unprocessedVariables), len(_backgroundAfterCuts)))
            
            for iVariable in _unprocessedVariables:
                _sortedSignal = np.sort(_signalDataFrame[iVariable].values)
                _sortedBackground = np.sort(_backgroundDataFrame[iVariable].values)
                #print(_sortedSignal)
                _tempSignificance, _tempCut = self.returnBestCutValue( iVariable, _sortedSignal, _sortedBackground, _method)
                #print ( iVariable, _tempSignificance, _tempCut )
                
                # most significant 1D variable so far in this iteration
                if _tempSignificance > _iBestSignificance:
                    _iBestSignificance = _tempSignificance
                    _iBestCut = _tempCut
                    _iBestVariable = iVariable
                    
            print('Iteration {0} chose variable {1} with significance {2} at cut {3}'.format(int(len(_varNames)-len(_unprocessedVariables)), _iBestVariable, _iBestSignificance, _iBestCut))
            _unprocessedVariables.remove(_iBestVariable)
            _orderedVariableAndCutDict[_iBestVariable] = [_iBestCut, _iBestSignificance]
            if 'mass' in _iBestVariable:
                _signalAfterCuts = _signalAfterCuts[ (_signalAfterCuts[_iBestVariable] > _iBestCut) & (_signalAfterCuts[_iBestVariable]< (_iBestCut + massWidth))]
                _backgroundAfterCuts = _backgroundAfterCuts[ (_backgroundAfterCuts[_iBestVariable] > _iBestCut) & (_backgroundAfterCuts[_iBestVariable]< (_iBestCut + massWidth))]
            else:
                _signalAfterCuts = _signalAfterCuts[ _signalAfterCuts[_iBestVariable] < _iBestCut ]
                _backgroundAfterCuts = _backgroundAfterCuts[ _backgroundAfterCuts[_iBestVariable] < _iBestCut ]
                
        return _orderedVariableAndCutDict


    def getCutsForSpecifiedEfficiency(self, _signal, _background, _eff, _inequality = '<'):
        """get cuts for all variables given user-specified efficency"""
        variablesAndCuts_ = {}
    
        print("========== Efficiency {0}% , Using {1} =========".format(_eff, _inequality) )
        for iColumn in range(0, len(_signal.columns) ):
            varName = self.variables[iColumn]
            sortedSignal = np.sort(_signal[varName].values)
            sortedBackground = np.sort(_background[varName].values)
            
            cutVal, nSig, nBkg = self.returnCutValueByConstantEfficiency( varName, sortedSignal, sortedBackground, _eff, _inequality)
            print('Cut of {4} {0} on {1} yields nSig = {2} and nBkg = {3}'.format(round(cutVal,2), varName, nSig, nBkg, _inequality))    
            variablesAndCuts_[varName] = round(cutVal,2)
    
        return variablesAndCuts_


    def returnNumberSignalAndBackgroundAfterCuts(self, _signal, _background, _cuts, _useVariableString = '', _inequality = '>', _significanceMetric = 'S/sqrt(B)'):
        """return number of signal and background (and maybe some significance score?) for a passed set of cuts"""
        
        _unprocessedVariables = [x for x in _cuts.keys() if _useVariableString in x] #list(_cuts.keys())
        _signalAfterCuts = _signal
        _backgroundAfterCuts = _background
        _nTotalSignal = len(_signalAfterCuts)
        _nTotalBackground = len(_backgroundAfterCuts)
        
        while len(_unprocessedVariables)>0:
            print('iteration {0}, signal has {1} rows'.format(len(_unprocessedVariables), len(_signalAfterCuts)))
            print('iteration {0}, background has {1} rows'.format(len(_unprocessedVariables), len(_backgroundAfterCuts)))
            
            for iVariable in _unprocessedVariables:
                _cutValue = _cuts[iVariable]
                _sortedSignal = np.sort(_signalAfterCuts[iVariable].values)
                _sortedBackground = np.sort(_backgroundAfterCuts[iVariable].values)
                
                _signalAfterCuts = _signalAfterCuts[ (_signalAfterCuts[iVariable] > _cutValue)]
                _backgroundAfterCuts = _backgroundAfterCuts[ (_backgroundAfterCuts[iVariable] > _cutValue)]
                
                _nSignal = len(_signalAfterCuts) * (_nTotalBackground / _nTotalSignal )
                _nBackground = len(_backgroundAfterCuts) 
                print('Iteration {0} chose variable {1} with N_signal = {2} ({4}) and N_background = {3} ({5})'.format(int(len(_unprocessedVariables)), iVariable, len(_signalAfterCuts), len(_backgroundAfterCuts), round(_nSignal,1), _nBackground))
                _unprocessedVariables.remove(iVariable)
                
        _nSignal = len(_signalAfterCuts) * (_nTotalBackground / _nTotalSignal )
        _nBackground = len(_backgroundAfterCuts) 
        print('{0} = {1}'.format(_significanceMetric, round( _nSignal / np.sqrt(_nBackground), 2)))
        
        return 
