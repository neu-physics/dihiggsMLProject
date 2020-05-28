#!/usr/bin/env python
#  Author:  Ben Tannenwald
##  Date:    April 16, 2020
##  Purpose: Class to load data, create user-defined CNN model, train CNN, evaluate performance, and save the trained model

# To-Do
#X) CM of phi for image?
#X) bb-center a la photography talk
#3) slide 9: remake these for diHiggs events that have the Higgs back-to-back ie no ISR. Are the blobs more pronounced than non-back-to-back diHiggs events?
#X) compare expected yields post-selection from this to other approaches. Much higher signal stats here! But by how much?
#5) subtract average a la Anna's idea
# ...
#X) compare ROC curves for different approaches a la Evan's work
#X) the pimples are weird in the ensemble QCD events. what causes the pimple?
#X) look at the per-jet-tag category yields after the macro-sample cut and compare to 4j4t S,B from other approaches
#9) do other processes matter?
#10) try those things you wanted


# Import the needed libraries
import os
import tensorflow as tf
import numpy as np
import matplotlib
#if "_CONDOR_SCRATCH_DIR" in os.environ:
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import h5py as h5

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, l1
from keras.initializers import Constant
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

#tf.logging.set_verbosity(tf.logging.FATAL) #where last text can be any of DEBUG, INFO, WARN, ERROR, or FATAL

seed = 7
np.random.seed(seed)

import sys
sys.path.insert(0, '/home/btannenw/Desktop/ML/dihiggsMLProject/')
sys.path.insert(0, '/uscms_data/d2/benjtann/ML/dihiggsMLProject/')
from utils.commonFunctions import *


def auc( y_true, y_pred ) :
      score = tf.py_func( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                          [y_true, y_pred],
                          'float32',
                          name='sklearnAUC' )
      return score


class cnnModelClass:
    
    def __init__ (self, _modelName, _cnnLayers, _ffnnLayers, _hhFile, _qcdFile, _imageCollection, _datasetPercentage, _loadSavedModel = False, _testRun = False, _condorRun=False, _useClassWeights=False, _extraVariables=[], _topDir='', _loadModelDir=''):
        self.modelName   = _modelName
        self.topDir = str(_topDir + '/' + _modelName + '/') if _topDir != '' else (_modelName + '/') 
        if not os.path.exists(self.topDir):
              os.makedirs(self.topDir)

        self.cnnLayers = _cnnLayers
        self.ffnnLayers = _ffnnLayers
        self.hhFile = _hhFile
        self.qcdFile = _qcdFile
        self.imageCollection = _imageCollection
        self.isTestRun = _testRun
        self.isCondorRun = _condorRun
        self.useClassWeights = _useClassWeights
        self.extraVariables = _extraVariables
        self.loadSavedModel = _loadSavedModel
        self.loadModelDir = str(_topDir + '/' + _loadModelName + '/') if _loadModelDir != '' else self.topDir
        self.datasetPercentage = _datasetPercentage
        self.runUNet = True #FIXME                

        # Class Defaults
        self.transparency = 0.88  # transparency of plots
        self.testingFraction = self.datasetPercentage/4
        self.nEventsForTesting = 1000        

        # Global Variables 
        self.hh = []
        self.qcd = []
        self.model = []
        self.best_model = []
        self.class_weights = []
        self.model_history = []
        self.images_train = []
        self.extraVariables_train = []
        self.labels_train = []
        self.images_test  = []
        self.extraVariables_test  = []
        self.labels_test  = []
        self.predictions_test  = []
        self.nQCD = -1
        self.nDihiggs = -1


        print("+++ Initialized {}".format(self.modelName) )
        
    ##############################################################
    ##                           MAIN                           ##
    ##############################################################

    def run(self):
      ##  Purpose: Class to load data, create user-defined CNN model, train CNN, evaluate performance, and save the trained model
          
      self.processInputs_v2() 
      #self.processInputs() 

      if self.runUNet:
            self.makeUNet() 
      else:
            if len(self.extraVariables) == 0:
                  self.makeCNN()
            else:
                  self.makeCNNPlus()
      
      if not self.loadSavedModel:
            self.trainCNN()
      
      self.evaluateModel()
                
      self.close()
        
      return

      
    ##############################################################
    ##                        functions                         ##
    ##############################################################

    def reinitialize(self,  _modelName, _cnnLayers, _ffnnLayers, _imageCollection, _loadSavedModel = False, _useClassWeights=False, _extraVariables=[], _loadModelDir=''):
        """ funtion to setup a new model instance while using data and some options from original configuration"""
        self.modelName   = _modelName
        _topDir = self.topDir.split('/')[0]
        self.topDir = str(_topDir + '/' + _modelName + '/') if _topDir != '' else (_modelName + '/') 
        if not os.path.exists(self.topDir):
              os.makedirs(self.topDir)

        self.cnnLayers = _cnnLayers
        self.ffnnLayers = _ffnnLayers
        self.imageCollection = _imageCollection
        self.useClassWeights = _useClassWeights
        self.loadSavedModel = _loadSavedModel
        self.loadModelDir = str(_topDir + '/' + _loadModelName + '/') if _loadModelDir != '' else self.topDir

        # Global Variables
        self.model = []
        self.best_model = []
        self.class_weights = []
        self.model_history = []
        self.predictions_test  = []

        print("+++ Initialized {}".format(self.modelName) )

        return
        
    def close(self):
        """ close things to save memory"""

        # Close files
        #self.hh.file.close()
        #self.qcd.file.close()
        
        return
    

    def processInputs(self):
      """ load images from files"""
      self.hh = [ [] for variable in range(0, len(self.extraVariables)+1) ] # index == 0 for images, 1-X are extra variables
      self.qcd = [ [] for variable in range(0, len(self.extraVariables)+1) ] # index == 0 for images, 1-X are extra variables
      
      if '.h5' in self.hhFile:     # ** A. User passed single .h5 file
            self.loadSingleFile( self.hhFile, isSignal = True)
      elif '.txt' in self.hhFile:  # ** B. User passed .txt with list of .h5 files
            self.loadMultipleFiles( self.hhFile, isSignal = True)
                
      if '.h5' in self.qcdFile:    # ** C. User single .h5 file
            self.loadSingleFile( self.qcdFile, isSignal = False)
      elif '.txt' in self.qcdFile: # ** D. User passed .txt with list of .h5 files
            self.loadMultipleFiles( self.qcdFile, isSignal = False)
            
      print("+ {} hh images, {} qcd images".format( len(self.hh[0]), len(self.qcd[0])))

      # Make labels
      hh_labels = np.ones( len(self.hh[0]) )
      qcd_labels = np.zeros( len(self.qcd[0]) )
      
      # Make combined dataset
      all_images = np.concatenate ( (self.hh[0], self.qcd[0]) )
      all_labels = np.concatenate ( (hh_labels.copy(), qcd_labels.copy()), axis=0)
      if 'composite' not in self.imageCollection:
            all_images = all_images.reshape( all_images.shape[0], all_images.shape[1], all_images.shape[2], 1)
      print("+ images shape: {}, labels shape:{}".format( all_images.shape, all_labels.shape) )

      # Make extra variables if needed
      if len(self.extraVariables) != 0:
            hh_extraVariables = np.transpose(np.stack( self.hh[1:].copy() ))
            qcd_extraVariables = np.transpose(np.stack( self.qcd[1:].copy() ))

            all_extraVariables = np.concatenate( (hh_extraVariables, qcd_extraVariables), axis=0)
            if len(all_extraVariables.shape) == 1:
                  all_extraVariables = all_extraVariables.reshape( all_extraVariables.shape[0], 1)
            print("+ extra variables shape: {}".format( all_extraVariables.shape))

            # ** Make output directory for saving scalers
            scaler_dir = os.path.join(self.topDir, "", "scalers")
            if not os.path.exists(scaler_dir):
                  os.makedirs(scaler_dir)
            # ** Do the scaling + saving
            scaler = MinMaxScaler()
            scaler.fit( all_extraVariables )
            all_extraVariables = scaler.transform( all_extraVariables )
            joblib.dump( scaler, '{}/mixMaxScaler.save'.format(scaler_dir)) 
            

      # Create training and testing sets
      if len(self.extraVariables) == 0: #images only
            self.images_train, self.images_test, self.labels_train, self.labels_test = train_test_split(all_images, all_labels, test_size=0.25, shuffle= True, random_state=30)
      else: # add extras
            self.images_train, self.images_test, self.extraVariables_train, self.extraVariables_test, self.labels_train, self.labels_test = train_test_split( all_images, all_extraVariables, all_labels, test_size=0.25, shuffle= True, random_state=30)
      return


    def processInputs_v2(self):
      """ load images from files"""
      self.hh = []  # index == 0 for images, 1-X are extra variables
      self.qcd = [] # index == 0 for images, 1-X are extra variables
      
      if '.h5' in self.hhFile:     # ** A. User passed single .h5 file
            _singleFile = self.hhFile.replace('/eos/uscms', 'root://cmseos.fnal.gov/') if self.isCondorRun else self.hhFile
            self.loadSingleFile_v2( _singeFile, isSignal = True)
      elif '.txt' in self.hhFile:  # ** B. User passed .txt with list of .h5 files
            self.loadMultipleFiles_v2( self.hhFile, isSignal = True)
                
      if '.h5' in self.qcdFile:    # ** C. User single .h5 file
            _singleFile = self.qcdFile.replace('/eos/uscms', 'root://cmseos.fnal.gov/') if self.isCondorRun else self.qcdFile
            self.loadSingleFile_v2( _singleFile, isSignal = False)
      elif '.txt' in self.qcdFile: # ** D. User passed .txt with list of .h5 files
            self.loadMultipleFiles_v2( self.qcdFile, isSignal = False)
            
      print("+ {} hh images, {} qcd images".format( len(self.hh), len(self.qcd)))
      maxEventsPerSample = min( len(self.hh), len(self.qcd))
      self.hh = self.hh[:maxEventsPerSample]
      self.qcd = self.qcd[:maxEventsPerSample]
      self.nDihiggs = len( self.hh )
      self.nQCD = len( self.qcd )
      print("+ {} hh images, {} qcd images".format( len(self.hh), len(self.qcd)))

      # Make combined image dataset
      all_images = np.concatenate ( (self.hh['compositeImages'], self.qcd['compositeImages']) )
      # -- reduction if only track/ecal/hcal images
      if 'trackImgs' in self.imageCollection:
            all_images = all_imagesa[:,:,:,0]
      elif 'nHadronImgs' in self.imageCollection:
            all_images = all_imagesa[:,:,:,1]
      elif 'photonImgs' in self.imageCollection:
            all_images = all_imagesa[:,:,:,2]

      # Make labels
      hh_labels = np.ones( len(self.hh) )
      qcd_labels = np.zeros( len(self.qcd) )
      all_labels = np.concatenate ( (hh_labels.copy(), qcd_labels.copy()), axis=0)
      if 'composite' not in self.imageCollection:
            all_images = all_images.reshape( all_images.shape[0], all_images.shape[1], all_images.shape[2], 1)
      print("+ images shape: {}, labels shape:{}".format( all_images.shape, all_labels.shape) )

      # Make extra variables if needed
      if len(self.extraVariables) != 0:
            hh_extraVariables = np.transpose( np.array([ self.hh[(var)] for var in self.extraVariables ].copy()) )
            qcd_extraVariables = np.transpose( np.array([ self.qcd[(var)] for var in self.extraVariables ].copy()) )

            all_extraVariables = np.concatenate( (hh_extraVariables, qcd_extraVariables), axis=0)
            if len(all_extraVariables.shape) == 1:
                  all_extraVariables = all_extraVariables.reshape( all_extraVariables.shape[0], 1)
            print("+ extra variables shape: {}".format( all_extraVariables.shape))

            # ** Make output directory for saving scalers
            scaler_dir = os.path.join(self.topDir, "", "scalers")
            if not os.path.exists(scaler_dir):
                  os.makedirs(scaler_dir)
            # ** Do the scaling + saving
            scaler = MinMaxScaler()
            scaler.fit( all_extraVariables )
            all_extraVariables = scaler.transform( all_extraVariables )
            joblib.dump( scaler, '{}/mixMaxScaler.save'.format(scaler_dir)) 
            

      # Create training and testing sets
      if len(self.extraVariables) == 0: #images only
            self.images_train, self.images_test, self.labels_train, self.labels_test = train_test_split(all_images, all_labels, test_size=0.25, shuffle= True, random_state=30)
      else: # add extras
            self.images_train, self.images_test, self.extraVariables_train, self.extraVariables_test, self.labels_train, self.labels_test = train_test_split( all_images, all_extraVariables, all_labels, test_size=0.25, shuffle= True, random_state=30)
      return

    
    def loadMultipleFiles(self, filename, isSignal):
          """ function for reading multiple files and adding to dataset"""
          
          if os.path.isfile(filename):
                # Do something with the file
                with open(filename, 'r') as f:
                      _fileList = f.readlines()
                      for _file in _fileList:
                            _singleFile = _file.split('\n')[0]
                            print( "Opening file: {} ...".format(_singleFile))
                            self.loadSingleFile( _singleFile, isSignal)          
          else:
                print("File not accessible")
                return

          return

    
    def loadSingleFile(self, filename, isSignal):
        """ function for reading single file and adding to dataset"""

        # open file
        _h5File = h5.File( filename, 'r' )

        # load slice of dataset
        _dataset = [ [] for variable in range(0, len(self.extraVariables)+1) ] # index == 0 for images, 1-X are extra variables

        if self.isTestRun:
            _dataset[0] = _h5File[ self.imageCollection ][:self.nEventsForTesting]
            for iExtra in range(0, len(self.extraVariables)):
                  _dataset[iExtra+1] = _h5File[ self.extraVariables[iExtra] ][:self.nEventsForTesting]
        else:
            _dataset[0] = _h5File[ self.imageCollection ][:]
            for iExtra in range(0, len(self.extraVariables)):
                  _dataset[iExtra+1] = _h5File[ self.extraVariables[iExtra] ][:]

        # store if signal
        if isSignal:
            # first file for this dataset
            if len(self.hh[0]) == 0:  
                self.hh[0] = _dataset[0]
                for iExtra in range(0, len(self.extraVariables)):
                      self.hh[iExtra+1] = _dataset[iExtra+1]
            # add file to existing dataset
            else:
                self.hh[0] = np.concatenate( (self.hh[0], _dataset[0]) )
                for iExtra in range(0, len(self.extraVariables)):
                      self.hh[iExtra+1] = np.concatenate( (self.hh[iExtra+1], _dataset[iExtra+1]) )

        # store if background
        else:
            # first file for this dataset
            if len(self.qcd[0]) == 0:  
                self.qcd[0] = _dataset[0]
                for iExtra in range(0, len(self.extraVariables)):
                      self.qcd[iExtra+1] = _dataset[iExtra+1]
            # add file to existing dataset
            else:
                self.qcd[0] = np.concatenate( (self.qcd[0], _dataset[0]) )
                for iExtra in range(0, len(self.extraVariables)):
                      self.qcd[iExtra+1] = np.concatenate( (self.qcd[iExtra+1], _dataset[iExtra+1]) )

        print( len(_dataset[0]), len(self.hh[0]), len(self.qcd[0]))

        # Close files
        _h5File.close()

        return

    def loadMultipleFiles_v2(self, filename, isSignal):
          """ function for reading multiple files and adding to dataset"""
          
          if os.path.isfile(filename):
                # Do something with the file
                with open(filename, 'r') as f:
                      _fileList = f.readlines()
                      for _file in _fileList:
                            _singleFile = _file.split('\n')[0]
                            _singleFile = _singleFile.replace('/eos/uscms', 'root://cmseos.fnal.gov/') if self.isCondorRun else _singleFile

                            print( "Opening file: {} ...".format(_singleFile))
                            self.loadSingleFile_v2( _singleFile, isSignal)          
          else:
                print("File not accessible")
                return

          return


    def loadSingleFile_v2(self, filename, isSignal):
        """ function for reading single file and adding to dataset"""

        # open file
        _h5File = h5.File( filename, 'r' )

        # load dataset (all or slice)
        _dataset = []

        if self.isTestRun:
            _dataset = _h5File['events'][:self.nEventsForTesting]
        else:
            _dataset = _h5File['events'][:]

        # create event mask
        _eventMask = self.returnEventMask(_dataset)
        _dataset = _dataset[_eventMask]

        # store if signal
        if isSignal:
            # first file for this dataset
            if len(self.hh) == 0:  
                self.hh = _dataset
            # add file to existing dataset
            else:
                self.hh = np.concatenate( (self.hh, _dataset))


        # store if background
        else:
            # first file for this dataset
            if len(self.qcd) == 0:  
                self.qcd = _dataset
            # add file to existing dataset
            else:
                self.qcd = np.concatenate( (self.qcd, _dataset) )

        print( len(_dataset), len(self.hh), len(self.qcd))

        # Close files
        _h5File.close()

        return

    def returnEventMask(self, _dataset):
          """return a mask for the dataset using case statement to determine criteria"""
          _nJetsMask = []
          _nBTagsMask = []
          _HTMask  = []

          # *** A. Mask for nJets
          if 'lessThan4j' in self.imageCollection:
                _nJetsMask = _dataset['nJets']<4
                print("**++ less than 4 jets")
          elif 'ge4j' in self.imageCollection:
                _nJetsMask = _dataset['nJets']>=4
                print("**++ greater than / equal to 4 jets")
          else:
                _nJetsMask = np.full( (len(_dataset)), True)  
                print("**++ no statement on Njets")

          # *** B. Mask for nBTags
          if '0b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']==0
          elif 'ge1b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']>=1
          elif '1b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']==1
          elif 'ge2b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']>=2
          elif '2b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']==2
          elif 'ge3b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']>=3
          elif '3b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']==3
          elif 'ge4b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']>=4
          elif '4b' in self.imageCollection:
                _nBTagsMask = _dataset['nBTags']==4
          else:
                _nBTagsMask = np.full( (len(_dataset)), True)  

          # *** B. Mask for HT
          if 'HT150' in self.imageCollection:
                _HTMask = _dataset['HT']>=150
          elif 'HT300' in self.imageCollection:
                _HTMask = _dataset['HT']>=300
          elif 'HT450' in self.imageCollection:
                _HTMask = _dataset['HT']>=450
          else:
                _HTMask = np.full( (len(_dataset)), True)  

          print("++** image collection: {}".format(self.imageCollection))
          print("++** Events passing HT mask: {}, events passing nJets mask: {}, events passing nBTags mask: {}".format( np.count_nonzero(_HTMask==True), np.count_nonzero(_nJetsMask==True), np.count_nonzero(_nBTagsMask==True)))
          _totalMask = np.logical_and.reduce( (_HTMask, _nBTagsMask, _nJetsMask) )
          print("++** Events all masks: {}".format( np.count_nonzero(_totalMask==True)))

          return _totalMask


    def singleEventImages(self, isSignal = False):
        """ make a few single event images"""

        if isSignal:
            imgs = self.hh[0]
        else:
            imgs = self.qcd[0]
            
        evt = 503
        plt.imshow(imgs[evt], alpha=self.transparency)
        plt.show()
        plt.imshow(imgs[evt+1], alpha=self.transparency)
        plt.show()
        plt.imshow(imgs[evt+2], alpha=self.transparency)
        plt.show()
        plt.imshow(imgs[evt+3], alpha=self.transparency)
        plt.clim(.5, 1)
        plt.show()

        return

    
    def makeCNN(self, loadBestModel = False):
        """ create CNN and return compiled model"""
        print("+++ Make CNN")

        # *** -1. Make directory if needed
        if (not os.path.exists(self.topDir)):
              print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(self.topDir))
              os.system("mkdir {0}".format(self.topDir))

        # *** 0. Set general options
        # ** A. General layer options. probably should allow for top-line flexibility here
        l2_reg = tf.keras.regularizers.l2(1e-4)
        conv_kwargs = dict(
            activation="relu",
            #kernel_initializer=tf.keras.initializers.lecun_normal(),
            #kernel_regularizer=l2_reg,
        )
        dense_kwargs = conv_kwargs

        # ** B. Intial output bias. probably should allow for top-line flexibility here
        initial_bias = np.log([len(self.hh[0])/len(self.qcd[0])])
        output_bias = Constant(initial_bias)
        print("+ initial bias: {}".format(initial_bias))
        
        # ** C. Create class weights for weighted training. probably should allow for top-line flexibility here
        # Scaling by total/2 helps keep the loss to a similar magnitude. The sum of the weights of all examples stays the same.
        if self.useClassWeights:
            total = len(self.qcd[0]) + len(self.hh[0])
            weight_for_0 = (1 / len(self.qcd[0]))*(total)/2.0 
            weight_for_1 = (1 / len(self.hh[0]))*(total)/2.0
            print('+ Weight for class 0: {:.2f}'.format(weight_for_0))
            print('+ Weight for class 1: {:.2f}'.format(weight_for_1))
            self.class_weights = {0: weight_for_0, 1: weight_for_1}        

        # *** 1. Define model
        _model = []
        _model = Sequential()
        pixelWidth = self.images_train.shape[1]
        #inputShape = self.images_train.shape[1:] if self.images_train.shape[-1] != pixelWidth else (self.images_train.shape[1:] + (1,))
        inputShape = self.images_train.shape[1:]
        print('++ input_shape (image) for CNN: {}'.format(inputShape) )

        # ** A. Convolutional component
        firstLayer = True
        for layer in self.cnnLayers:
            # layer format example: [ "Conv2D", [16, (3,3)]] or ["MaxPooling2D", [(3, 3)]]
            if layer[0] == "Conv2D" and firstLayer:
                _model.add( Conv2D( layer[1][0], layer[1][1], input_shape= inputShape, **conv_kwargs) )
                firstLayer = False
            elif layer[0] == "Conv2D" and not firstLayer:
                _model.add( Conv2D( layer[1][0], layer[1][1], **conv_kwargs) )
            elif layer[0] == "MaxPooling2D":
                _model.add( MaxPooling2D( layer[1][0]) )

        # ** B. Flatten model for input to feed-forward network
        _model.add( Flatten() ) 

        # ** C. Feed-forward component
        for layer in self.ffnnLayers:
            # layer format example: [ "Dense", [64]] or ["BatchNormalization"], ["Dropout", [0.2]]
            if layer[0] == "Dense":
                _model.add( Dense( layer[1][0], **dense_kwargs) )
            elif layer[0] == "BatchNormalization":
                _model.add(  BatchNormalization() )
            elif layer[0] == "Dropout":
                _model.add( Dropout( layer[1][0]) )

        
        # ** D. Output layer
        _model.add( Dense(1, activation='sigmoid', bias_initializer=output_bias) )
    
        # ** E. Print summary
        print("++ Model Summary\n {}".format(_model.summary()))

        # ** F. Define metrics and compile
        metrics = [ tf.keras.metrics.categorical_accuracy,
                    'accuracy',
                    #tf.keras.metrics.AUC(name='auc'),
                    auc,
        ]

        _model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=metrics,
        )

        # ** G. Load model if appropriate
        if self.loadSavedModel or loadBestModel:
            local_dir = os.path.join(self.loadModelDir, "models")
            modelfile = os.path.join(local_dir, self.modelName)+'.hdf5'
            print("++ loading model from {}".format(modelfile))
            if not os.path.isfile(modelfile):
                print("--- ERROR, Modelfile {} NOT FOUND. No weights initialized".format(modefile))
                return
        
            loadShape = (1,) + inputShape
            _model.predict( np.empty( loadShape ) )
            _model.load_weights(modelfile)
            self.best_model = _model
        else:
            self.model = _model

            
        return


    def makeCNNPlus(self, loadBestModel = False):
        """ create CNN that uses additional inputs after convolution and return compiled model"""
        print("+++ Make CNN")

        # *** -1. Make directory if needed
        if (not os.path.exists(self.topDir)):
              print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(self.topDir))
              os.system("mkdir {0}".format(self.topDir))

        # *** 0. Set general options
        # ** A. General layer options. probably should allow for top-line flexibility here
        l2_reg = tf.keras.regularizers.l2(1e-4)
        conv_kwargs = dict(
            activation="relu",
            #kernel_initializer=tf.keras.initializers.lecun_normal(),
            #kernel_regularizer=l2_reg,
        )
        dense_kwargs = conv_kwargs

        # ** B. Intial output bias. probably should allow for top-line flexibility here
        initial_bias = np.log([len(self.hh[0])/len(self.qcd[0])])
        output_bias = Constant(initial_bias)
        print("+ initial bias: {}".format(initial_bias))
        
        # ** C. Create class weights for weighted training. probably should allow for top-line flexibility here
        # Scaling by total/2 helps keep the loss to a similar magnitude. The sum of the weights of all examples stays the same.
        if self.useClassWeights:
            total = len(self.qcd[0]) + len(self.hh[0])
            weight_for_0 = (1 / len(self.qcd[0]))*(total)/2.0 
            weight_for_1 = (1 / len(self.hh[0]))*(total)/2.0
            print('+ Weight for class 0: {:.2f}'.format(weight_for_0))
            print('+ Weight for class 1: {:.2f}'.format(weight_for_1))
            self.class_weights = {0: weight_for_0, 1: weight_for_1}        

        # *** 1. Define model
        pixelWidth = self.images_train.shape[1]
        #inputShape = self.images_train.shape[1:] if self.images_train.shape[-1] != pixelWidth else (self.images_train.shape[1:] + (1,))
        inputShape = self.images_train.shape[1:]
        print('++ input_shape (image) for CNN: {}'.format(inputShape) )

        # ** A. Convolutional component
        firstLayer = True
        convInput  = Input(shape=inputShape)
        convNN = []
        for layer in self.cnnLayers:
            # layer format example: [ "Conv2D", [16, (3,3)]] or ["MaxPooling2D", [(3, 3)]]
            if layer[0] == "Conv2D" and firstLayer:
                  convNN = Conv2D( layer[1][0], layer[1][1], **conv_kwargs)(convInput)
                  firstLayer = False
            elif layer[0] == "Conv2D" and not firstLayer:
                  convNN = Conv2D( layer[1][0], layer[1][1], **conv_kwargs)(convNN)
            elif layer[0] == "MaxPooling2D":
                  convNN = MaxPooling2D( layer[1][0])(convNN)

        # ** B. Flatten model for input to feed-forward network
        flattenInput = Input(shape=convNN.shape)
        convNN = Flatten()( convNN )
        
        # ** C. Extra information to add to flattened CNN output
        extraInput = Input(shape=self.extraVariables_train.shape[1:])
        extra = Activation('linear')(extraInput)

        # ** D. Concatenate extra+CNN as input for FC layers
        concatenated = concatenate([convNN, extra])

        # ** E. Feed-forward components
        ffNN = []
        firstLayer = True
        for layer in self.ffnnLayers:
            # layer format example: [ "Dense", [64]] or ["BatchNormalization"], ["Dropout", [0.2]]
            if layer[0] == "Dense":
                  if firstLayer:
                        ffNN = Dense( layer[1][0], **dense_kwargs)(concatenated)
                        firstLayer = False
                  else:
                        ffNN = Dense( layer[1][0], **dense_kwargs)(ffNN)
            elif layer[0] == "BatchNormalization":
                  ffNN = BatchNormalization()(ffNN)
            elif layer[0] == "Dropout":
                  ffNN = Dropout( layer[1][0])(ffNN)

        # ** F. Output layer
        ffNN = Dense(1, activation='sigmoid', bias_initializer=output_bias)(ffNN)

        # ** G. Put it all together
        _model = Model([convInput, extraInput], ffNN)
      
        # ** H. Print summary
        print("++ Model Summary\n {}".format(_model.summary()))

        # ** I. Define metrics and compile
        metrics = [ tf.keras.metrics.categorical_accuracy,
                    'accuracy',
                    #tf.keras.metrics.AUC(name='auc'),
                    auc,
        ]

        _model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=metrics,
        )

        # ** J. Load model if appropriate
        if self.loadSavedModel or loadBestModel:
            local_dir = os.path.join(self.loadModelDir, "models")
            modelfile = os.path.join(local_dir, self.modelName)+'.hdf5'
            print("++ loading model from {}".format(modelfile))
            if not os.path.isfile(modelfile):
                print("--- ERROR, Modelfile {} NOT FOUND. No weights initialized".format(modefile))
                return
        
            loadShape = np.empty( (1,) + inputShape )
            extraShape = np.empty( (1,) + self.extraVariables_train.shape[1:] )

            _model.predict( [loadShape, extraShape] )
            _model.load_weights(modelfile)
            self.best_model = _model
        else:
            self.model = _model

            
        return


    def makeUNet(self, loadBestModel = False):
        """ create U-Net that makes runs over multiple uses additional inputs after convolution and return compiled model"""
        print("+++ Make CNN")

        # *** -1. Make directory if needed
        if (not os.path.exists(self.topDir)):
              print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(self.topDir))
              os.system("mkdir {0}".format(self.topDir))

        # *** 0. Set general options
        # ** A. General layer options. probably should allow for top-line flexibility here
        l2_reg = tf.keras.regularizers.l2(1e-4)
        conv_kwargs = dict(
            activation="relu",
            #kernel_initializer=tf.keras.initializers.lecun_normal(),
            #kernel_regularizer=l2_reg,
        )
        dense_kwargs = conv_kwargs

        # ** B. Intial output bias. probably should allow for top-line flexibility here
        initial_bias = np.log([len(self.hh[0])/len(self.qcd[0])])
        output_bias = Constant(initial_bias)
        print("+ initial bias: {}".format(initial_bias))
        
        # ** C. Create class weights for weighted training. probably should allow for top-line flexibility here
        # Scaling by total/2 helps keep the loss to a similar magnitude. The sum of the weights of all examples stays the same.
        if self.useClassWeights:
            total = len(self.qcd[0]) + len(self.hh[0])
            weight_for_0 = (1 / len(self.qcd[0]))*(total)/2.0 
            weight_for_1 = (1 / len(self.hh[0]))*(total)/2.0
            print('+ Weight for class 0: {:.2f}'.format(weight_for_0))
            print('+ Weight for class 1: {:.2f}'.format(weight_for_1))
            self.class_weights = {0: weight_for_0, 1: weight_for_1}        

        # *** 1. Define model
        pixelWidth = self.images_train.shape[1]
        #inputShape = self.images_train.shape[1:] if self.images_train.shape[-1] != pixelWidth else (self.images_train.shape[1:] + (1,))
        inputShape = self.images_train.shape[1:]
        print('++ input_shape (image) for CNN: {}'.format(inputShape) )

        # ** A. "Usual" Convolutional component
        firstLayer = True
        convInput  = Input(shape=inputShape)
        convNN = []
        for layer in self.cnnLayers:
            # layer format example: [ "Conv2D", [16, (3,3)]] or ["MaxPooling2D", [(3, 3)]]
            if layer[0] == "Conv2D" and firstLayer:
                  convNN = Conv2D( layer[1][0], layer[1][1], **conv_kwargs)(convInput)
                  firstLayer = False
            elif layer[0] == "Conv2D" and not firstLayer:
                  convNN = Conv2D( layer[1][0], layer[1][1], **conv_kwargs)(convNN)
            elif layer[0] == "MaxPooling2D":
                  convNN = MaxPooling2D( layer[1][0])(convNN)

        # ** B. Flatten model for input to feed-forward network
        flattenInput = Input(shape=convNN.shape)
        convNN = Flatten()( convNN )
        
        # ** C1. Parallel convolutional components (the "U")
        wholeImageUinput = Input(shape=inputShape)
        #wholeImageU = Conv2D( 16, (pixelWidth, pixelWidth), **conv_kwargs)(convInput)
        wholeImageU = Conv2D( 1, (pixelWidth, pixelWidth), **conv_kwargs)(convInput)
        #wholeImageU = Conv2D( 3, (pixelWidth, pixelWidth), **conv_kwargs)(convInput)
        wholeImageU = Flatten()( wholeImageU )

        # ** C2. Parallel convolutional components (the "U")
        halfSize = int( np.ceil( pixelWidth/2) )
        quadrantImageU = Conv2D( 4, (halfSize, halfSize), **conv_kwargs)(convInput)
        ##quadrantImageU = Conv2D( 16, (halfSize, halfSize), **conv_kwargs)(convInput)
        quadrantImageU = Flatten()( quadrantImageU )

        # ** D. Concatenate extra+CNN as input for FC layers
        #concatenated = concatenate([convNN, wholeImageU])
        concatenated = concatenate([convNN, wholeImageU, quadrantImageU])


        # ** E. Feed-forward components
        ffNN = []
        firstLayer = True
        for layer in self.ffnnLayers:
            # layer format example: [ "Dense", [64]] or ["BatchNormalization"], ["Dropout", [0.2]]
            if layer[0] == "Dense":
                  if firstLayer:
                        ffNN = Dense( layer[1][0], **dense_kwargs)(concatenated)
                        firstLayer = False
                  else:
                        ffNN = Dense( layer[1][0], **dense_kwargs)(ffNN)
            elif layer[0] == "BatchNormalization":
                  ffNN = BatchNormalization()(ffNN)
            elif layer[0] == "Dropout":
                  ffNN = Dropout( layer[1][0])(ffNN)

        # ** F. Output layer
        ffNN = Dense(1, activation='sigmoid', bias_initializer=output_bias)(ffNN)

        # ** G. Put it all together
        #_model = Model([convInput, extraInput], ffNN)
        _model = Model(convInput, ffNN)
      
        # ** H. Print summary
        print("++ Model Summary\n {}".format(_model.summary()))

        # ** I. Define metrics and compile
        metrics = [ tf.keras.metrics.categorical_accuracy,
                    'accuracy',
                    #tf.keras.metrics.AUC(name='auc'),
                    auc,
        ]

        _model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=metrics,
        )

        # ** J. Load model if appropriate
        if self.loadSavedModel or loadBestModel:
            local_dir = os.path.join(self.loadModelDir, "models")
            modelfile = os.path.join(local_dir, self.modelName)+'.hdf5'
            print("++ loading model from {}".format(modelfile))
            if not os.path.isfile(modelfile):
                print("--- ERROR, Modelfile {} NOT FOUND. No weights initialized".format(modefile))
                return
        
            loadShape = np.empty( (1,) + inputShape )

            _model.predict( [loadShape] )
            _model.load_weights(modelfile)
            self.best_model = _model
        else:
            self.model = _model

            
        return


    def trainCNN(self):
      print("+++ Train CNN" )
      
      # *** 0. Set GPU limit
      #gpus = tf.config.experimental.list_physical_devices('GPU')
      #if gpus:
      #      # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
      #      try:
      #            tf.config.experimental.set_virtual_device_configuration(
      #                  gpus[0],
      #                  [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
      #                 )
      #            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      #            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      #      except RuntimeError as e:
      #            # Virtual devices must be set before GPUs have been initialized
      #            print(e)

      # *** 1. Define output directory
      model_dir = os.path.join(self.topDir, "", "models")
      if not os.path.exists(model_dir):
            os.makedirs(model_dir)
      
      # *** 2. Define callbacks for training
      fit_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                  filepath=os.path.join(model_dir, self.modelName)+'.hdf5',
                  save_best_only=True,
                  save_weights_only=True,
                  monitor="val_auc",
                  mode="max",
                  #monitor="val_loss",
                  #mode="min",
            ),
            tf.keras.callbacks.EarlyStopping(
                  monitor="val_auc",
                  mode="max",
                  #monitor='val_loss', 
                  #mode='min', 
                  verbose=1, 
                  patience=15,  
                  min_delta=.0025,
            ),
      ]
        
      
      # *** 3. Train model
      nEpochs = 10 if self.isTestRun else 50
      trainOpts = dict(shuffle=True,
                       epochs=nEpochs, 
                       batch_size=2056, #512,
                       callbacks=fit_callbacks,
                 )
      if self.useClassWeights:
            trainOpts['class_weight']=self.class_weights
            
            
      if len(self.extraVariables) == 0:
            self.model_history = self.model.fit(self.images_train, self.labels_train,
                                                validation_data=(self.images_test, self.labels_test),
                                                **trainOpts,
                                          )
      else:
            self.model_history = self.model.fit( [self.images_train, self.extraVariables_train], self.labels_train,
                                                 validation_data=([self.images_test, self.extraVariables_test], self.labels_test),
                                                 **trainOpts,
                                           )

            
      return
    



    def evaluateModel(self):
        """ evaluateModel( modelNoWeights, historyNoWeights, self.images_test, self.labels_test) """
        print("+++ Evaluate Model")
        
        # *** 0. Use best/loaded model for evaluation
        if not self.loadSavedModel:
              if self.runUNet:
                    self.makeUNet( loadBestModel= True)
              else:
                    if len(self.extraVariables) == 0:
                          self.makeCNN( loadBestModel = True )
                    else:
                          self.makeCNNPlus( loadBestModel=True)

                    

        # *** 1. set inputs --> special treatment if loading saved model
        images = []
        labels = []
        if self.loadSavedModel:
            images = np.concatenate ( (self.images_test.copy(), self.images_train.copy()), axis=0)
            extras = np.concatenate ( (self.extraVariables_test.copy(), self.extraVariables_train.copy()), axis=0)
            labels = np.concatenate ( (self.labels_test.copy(), self.labels_train.copy()), axis=0)
            images, labels = shuffle(images, labels, random_state=0)
        else:
            images = self.images_test
            extras = self.extraVariables_test
            labels = self.labels_test
        
        # *** 2. Make prediction data
        hh_data_test    = np.asarray([x for x,y in zip( images, labels) if y==1])
        hh_extras_test  = np.asarray([x for x,y in zip( extras, labels) if y==1])
        hh_labels_test  = np.asarray([y for x,y in zip( images, labels) if y==1])
        qcd_data_test   = np.asarray([x for x,y in zip( images, labels) if y==0])
        qcd_extras_test = np.asarray([x for x,y in zip( extras, labels) if y==0])
        qcd_labels_test = np.asarray([y for x,y in zip( images, labels) if y==0])
                
        # *** 3. Make history plots
        if not self.loadSavedModel:
            print("++ Training History Plots")
            makeHistoryPlots( self.model_history, ['acc', 'loss', 'auc'], self.modelName, savePlot=True, saveDir=self.topDir )
        
        # *** 4. Make predictions
        #print("++ Calculate scores")
        # ** A. Approach, the first
        #score_hh = self.best_model.evaluate(hh_data_test, hh_labels_test)
        #score_qcd = self.best_model.evaluate(qcd_data_test, qcd_labels_test)
        # ** B. Approach, the second
        #scores = self.best_model.evaluate(images, labels)
        #score_hh = [score for score,label in zip(scores, labels) if label==1]
        #score_qcd = [score for score,label in zip(scores, labels) if label==0]
        #print(score_hh, score_qcd)
        print("++ Make predictions")
        if len(self.extraVariables)==0:
              pred_hh = self.best_model.predict(hh_data_test)
              pred_qcd = self.best_model.predict(qcd_data_test)
        else:
              pred_hh = self.best_model.predict( [hh_data_test, hh_extras_test] )
              pred_qcd = self.best_model.predict( [qcd_data_test, qcd_extras_test] )

        # *** 5. make output score plot
        print("++ Output Score Plot")
        _nBins = 40
        predictionResults = {'hh_pred':pred_hh, 'qcd_pred':pred_qcd}
        #sig, cut, sigErr = compareManyHistograms( predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'CNN Score', 0, 1, _nBins, _yMax = 5, _normed=True, savePlot=True, saveDir=self.topDir, writeSignificance=True, _testingFraction=self.testingFraction )
#sig, cut, sigErr = compareManyHistograms( predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'CNN Score', 0, 1, _nBins, _yMax = 5, _normed=True, savePlot=True, saveDir=self.topDir, writeSignificance=True, nDihiggs=self.nDihiggs, nQCD=self.nQCD )       
        sig, cut, sigErr = compareManyHistograms( predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'CNN Score', 0, 1, _nBins, _yMax = 5, _normed=True, savePlot=True, saveDir=self.topDir, writeSignificance=True, _testingFraction=self.testingFraction, nDihiggs=self.nDihiggs, nQCD=self.nQCD )
      
        # *** 6. Get best cut value for CNN assuming some minimal amount of signal
        #pred_hh_sig = [x[0] for x in pred_hh.copy()]
        #pred_qcd_sig = [x[0] for x in pred_qcd.copy()]
        #sig, cut, sigErr = returnBestCutValue('CNN', pred_hh_sig, pred_qcd_sig, _minBackground=400e3, _testingFraction=self.testingFraction)

        ## *** 6B. Get signifiance for any user-specified NN score cut value
        #testingFraction = 0.1
        #lumiscale_hh  = getLumiScaleFactor(testingFraction, True, 25e3)
        #lumiscale_qcd = getLumiScaleFactor(testingFraction, False, 50e3)
        #cut = 0.48
        #_nSignal = sum( value > cut for value in pred_hh_sig)*lumiscale_hh
        #_nBackground = sum( value > cut for value in pred_qcd_sig)*lumiscale_qcd
        #
        #print('nSig = {0} , nBkg = {1} with significance = {2} for NN score > {3}'.format(_nSignal, _nBackground, _nSignal/np.sqrt(_nBackground), cut) )

        # *** 7. Confusion matrix
        print("++ Confusion Matrix")
        if len(self.extraVariables)==0:
              self.predictions_test = self.best_model.predict(self.images_test)
        else:
              self.predictions_test = self.best_model.predict( [self.images_test, self.extraVariables_test] )
        cm = confusion_matrix( self.labels_test, self.predictions_test > cut)
        print(cm)
        print('QCD called QCD (True Negatives): {} ({}%)'.format( cm[0][0], round(100*(cm[0][0]/sum(cm[0]))) ))
        print('QCD called Dihiggs (False Positives):  {} ({}%)'.format( cm[0][1], round(100*(cm[0][1]/sum(cm[0]))) ))
        print('Dihiggs called QCD (False Negatives):  {} ({}%)'.format( cm[1][0], round(100*(cm[1][0]/sum(cm[1]))) ))
        print('Dihiggs called Dihiggs (True Positives):  {} ({}%)'.format( cm[1][1], round(100*(cm[1][1]/sum(cm[1]))) ))
        print('Total Dihiggs: ', np.sum(cm[1]))
        
        # *** 8. Make ROC curve
        print("++ ROC Curve")
        makeEfficiencyCurves( dict(label="CNN", labels=self.labels_test, prediction=self.predictions_test, color="blue"), _modelName = self.modelName, savePlot=True, saveDir=self.topDir)
        #utils/commonFunctions.py: def makeEfficiencyCurves(*data, _modelName='', savePlot=False, saveDir=''):

        return


#==========================================================================================



## *** 7. Make overlaid ROC curves
#predsNoWeights = modelNoWeights.predict(self.images_test)
#predsWeighted  = modelWeighted.predict(self.images_test)
#overlay = [ dict(label="No Weights", labels = model_1.labels_test, prediction = model_1.predictions_test, color="blue"),
#            dict(label="Weighted", labels = model_2.labels_test, prediction = model_2.predictions_test, color="red")
#          ]
#overlayROCCurves( overlay )





