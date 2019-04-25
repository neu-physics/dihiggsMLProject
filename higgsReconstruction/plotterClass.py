##  Author:  Ben Tannenwald
##  Date:    April 3, 2019
##  Purpose: Class to hold functions for plotting stuff

import os
import matplotlib.pyplot as plt

class hepPlotter:
    def __init__ (self, _subdirectoryName):
        # Class Defaults
        self.transparency = 0.5  # transparency of plots
        self.subdirectory = _subdirectoryName

        if os.path.isdir( self.subdirectory )==False:
            os.mkdir( self.subdirectory )

    def setTransparency(self, _userTransparency):
        self.transparency = _userTransparency
    def getTransparency(self):
        print ("Transparency: ", self.transparency)


    ##############################################################
    ##                FUNCTIONS FOR PLOTTING                    ##
    ##############################################################

    def compareManyHistograms(self, _pairingAlgorithm, _labels, _nPlot, _title, _xtitle, _xMin, _xMax, _nBins, _normed=False):
        #_mean_arrAll     = np.mean(_arrAll)
        #_stdev_arrAll    = np.std(_arrAll)
        #_nEntries_arrAll = len(_arrAll)
        #s1 = _xtitle + ':Entries = {0}, mean = {1:4F}, std dev = {2:4f}\n'.format(_nEntries_arrAll, _mean_arrAll, _stdev_arrAll)
        
        if len( self.plottingData[_pairingAlgorithm].keys()) < len(_labels):
            print ("!!! Unequal number of arrays and labels. Learn to count better.")
            return 0
    
        plt.figure(_nPlot)
        if _normed:
            plt.title(_title + ' (Normalized)')
        else:
            plt.title(_title)
        plt.xlabel(_xtitle)
        _bins = np.linspace(_xMin, _xMax, _nBins)
   
        for iLabel in _labels:
            plt.hist(self.plottingData[_pairingAlgorithm][iLabel], _bins, alpha=self.transparency, normed=_normed, label= iLabel+' Events')
        plt.legend(loc='upper right')
        #plt.text(.1, .1, s1)
    
        # store figure copy for later saving
        fig = plt.gcf()
    
        # draw interactively
        plt.show()
    
        # save an image files
        _scope    = _title.split(' ')[0].lower()
        _variable = _xtitle.lstrip('Jet Pair').replace(' ','').replace('[GeV]','')
        _allLabels = ''.join(_labels)
        _filename  = self.subdirectory + '/' + _scope + '_' + pairingAlgorithm + '_' + _allLabels + '_' + _variable
        if _normed:
            _filename = _filename + '_norm'
        fig.savefig( _filename+'.png' )

        return
