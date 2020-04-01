##  Author:  Ben Tannenwald
##  Date:    April 1, 2020
##  Purpose: Class to handle making images for CNN consumption from jet constituent data

import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import h5py as h5

import os 
import uproot, uproot_methods
import uproot_methods.classes.TLorentzVector as TLorentzVector
import uproot_methods.classes.TVector3 as TVector3

import sys
sys.path.insert(0, '/home/btannenw/Desktop/ML/dihiggsMLProject/higgsReconstruction/')
from JetCons import JetCons


class imageMaker:

    def __init__ (self, _datasetName, _inputFile, _isTestRun):
        self.datasetName = _datasetName
        self.inputFileName = _inputFile
        self.isTestRun = _isTestRun
        if os.path.isdir( self.datasetName )==False:
            os.mkdir( self.datasetName )

        # Class Defaults
        self.tracks = [ ]
        self.nHadrons = [ ]
        self.photons = [ ]


    ##############################################################
    ##                           MAIN                           ##
    ##############################################################

    def makeImages(self):
        """ central function call body for producing images"""

        
