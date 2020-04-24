#!/usr/bin/env python
##  Author:  Ben Tannenwald
##  Date:    April 16, 2020
##  Purpose: Class to load data, create user-defined CNN model, train CNN, evaluate performance, and save the trained model


# To-Do
#X) CM of phi for image?
#2) bb-center a la photography talk
#3) slide 9: remake these for diHiggs events that have the Higgs back-to-back ie no ISR. Are the blobs more pronounced than non-back-to-back diHiggs events?
#X) compare expected yields post-selection from this to other approaches. Much higher signal stats here! But by how much?
#5) subtract average a la Anna's idea
# ...
#X) compare ROC curves for different approaches a la Evan's work
#7) the pimples are weird in the ensemble QCD events. what causes the pimple?
#8) look at the per-jet-tag category yields after the macro-sample cut and compare to 4j4t S,B from other approaches
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


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, l1
from keras.initializers import Constant
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
    
    def __init__ (self, _modelName, _cnnLayers, _ffnnLayers, _hhFile, _qcdFile, _imageCollection, _datasetPercentage, _loadSavedModel = False, _testRun = False, _useClassWeights=False, _topDir='', _loadModelDir=''):
        self.modelName   = _modelName
        self.topDir = str(_topDir + '/' + _modelName + '/') if _topDir != '' else (_modelName + '/') 
        self.cnnLayers = _cnnLayers
        self.ffnnLayers = _ffnnLayers
        self.hhFile = _hhFile
        self.qcdFile = _qcdFile
        self.imageCollection = _imageCollection
        self.isTestRun = _testRun
        self.useClassWeights = _useClassWeights
        self.loadSavedModel = _loadSavedModel
        self.loadModelDir = str(_topDir + '/' + _loadModelName + '/') if _loadModelDir != '' else self.topDir
        self.datasetPercentage = _datasetPercentage
        
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
        self.labels_train = []
        self.images_test  = []
        self.labels_test  = []
        self.predictions_test  = []


        print("+++ Initialized {}".format(self.modelName) )
        
    ##############################################################
    ##                           MAIN                           ##
    ##############################################################

    def run(self):
      ##  Purpose: Class to load data, create user-defined CNN model, train CNN, evaluate performance, and save the trained model
          
      self.processInputs() 
          
      self.makeCNN()
          
      if not self.loadSavedModel:
            self.trainCNN()
      
      self.evaluateModel()
                
      self.close()
        
      return

      
    ##############################################################
    ##                        functions                         ##
    ##############################################################

    def reinitialize(self,  _modelName, _cnnLayers, _ffnnLayers, _imageCollection, _loadSavedModel = False, _useClassWeights=False, _loadModelDir=''):
        """ funtion to setup a new model instance while using data and some options from original configuration"""
        self.modelName   = _modelName
        _topDir = self.topDir.split('/')[0]
        self.topDir = str(_topDir + '/' + _modelName + '/') if _topDir != '' else (_modelName + '/') 
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
        self.hh = []
        self.qcd = []
        
        if '.h5' in self.hhFile:     # ** A. User passed single .h5 file
            self.loadSingleFile( self.hhFile, isSignal = True)
        elif '.txt' in self.hhFile:  # ** B. User passed .txt with list of .h5 files
            self.loadMultipleFiles( self.hhFile, isSignal = True)
        
        if '.h5' in self.qcdFile:    # ** C. User single .h5 file
            self.loadSingleFile( self.qcdFile, isSignal = False)
        elif '.txt' in self.qcdFile: # ** D. User passed .txt with list of .h5 files
            self.loadMultipleFiles( self.qcdFile, isSignal = False)

        print("+ {} hh images, {} qcd images".format( len(self.hh), len(self.qcd)))

        # Make labels
        hh_labels = np.ones( len(self.hh) )
        qcd_labels = np.zeros( len(self.qcd) )
        
        # Make combined dataset
        all_images = np.concatenate ( (self.hh, self.qcd) )
        if 'composite' not in self.imageCollection:
              all_images = all_images.reshape( all_images.shape[0], all_images.shape[1], all_images.shape[2], 1)
        all_labels = np.concatenate ( (hh_labels.copy(), qcd_labels.copy()), axis=0)
        print("+ images shape: {}, labels shape:{}".format( all_images.shape, all_labels.shape) )

        # Create training and testing sets
        self.images_train, self.images_test, self.labels_train, self.labels_test = train_test_split(all_images, all_labels, test_size=0.25, shuffle= True, random_state=30)

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
        _dataset = []
        if self.isTestRun:
            _dataset = _h5File[ self.imageCollection ][:self.nEventsForTesting]
        else:
            _dataset = _h5File[ self.imageCollection ][:]

        # store if signal
        if isSignal:
            # first file for this dataset
            if len(self.hh) == 0:  
                self.hh = _dataset
            # add file to existing dataset
            else:
                self.hh = np.concatenate( (self.hh, _dataset) )
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

    
    def singleEventImages(self, isSignal = False):
        """ make a few single event images"""

        if isSignal:
            imgs = self.hh
        else:
            imgs = self.qcd
            
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
        initial_bias = np.log([len(self.hh)/len(self.qcd)])
        output_bias = Constant(initial_bias)
        print("+ initial bias: {}".format(initial_bias))
        
        # ** C. Create class weights for weighted training. probably should allow for top-line flexibility here
        # Scaling by total/2 helps keep the loss to a similar magnitude. The sum of the weights of all examples stays the same.
        if self.useClassWeights:
            total = len(self.qcd) + len(self.hh)
            weight_for_0 = (1 / len(self.qcd))*(total)/2.0 
            weight_for_1 = (1 / len(self.hh))*(total)/2.0
            print('+ Weight for class 0: {:.2f}'.format(weight_for_0))
            print('+ Weight for class 1: {:.2f}'.format(weight_for_1))
            self.class_weights = {0: weight_for_0, 1: weight_for_1}        

        # *** 1. Define model
        _model = []
        _model = Sequential()
        pixelWidth = self.images_train.shape[1]
        #inputShape = self.images_train.shape[1:] if self.images_train.shape[-1] != pixelWidth else (self.images_train.shape[1:] + (1,))
        inputShape = self.images_train.shape[1:]
        print('++ input_shape for CNN: {}'.format(inputShape) )

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


    def trainCNN(self):
        print("+++ Train CNN" )

        # *** 1. Define output directory
        topDir = self.topDir
        name = self.modelName
        if not os.path.exists(topDir):
            os.makedirs(topDir)
        model_dir = os.path.join(topDir, "", "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
      
        # *** 2. Define callbacks for training
        fit_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, name)+'.hdf5',
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
        
        self.model_history = self.model.fit(self.images_train, self.labels_train,
                                            validation_data=(self.images_test, self.labels_test),
                                            **trainOpts,
        )

        
        return
    


    def evaluateModel(self):
        """ evaluateModel( modelNoWeights, historyNoWeights, self.images_test, self.labels_test) """
        print("+++ Evaluate Model")
        
        # *** 0. Use best/loaded model for evaluation
        if not self.loadSavedModel:
            self.makeCNN( loadBestModel = True )

        # *** 1. set inputs --> special treatment if loading saved model
        images = []
        labels = []
        if self.loadSavedModel:
            images = np.concatenate ( (self.images_test.copy(), self.images_train.copy()), axis=0)
            labels = np.concatenate ( (self.labels_test.copy(), self.labels_train.copy()), axis=0)
            images, labels = shuffle(images, labels, random_state=0)
        else:
            images = self.images_test
            labels = self.labels_test
        
        # *** 2. Make prediction data
        hh_data_test    = np.asarray([x for x,y in zip( images, labels) if y==1])
        hh_labels_test  = np.asarray([y for x,y in zip( images, labels) if y==1])
        qcd_data_test   = np.asarray([x for x,y in zip( images, labels) if y==0])
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
        pred_hh = self.best_model.predict(hh_data_test)
        pred_qcd = self.best_model.predict(qcd_data_test)
      
        # *** 5. make output score plot
        print("++ Output Score Plot")
        _nBins = 40
        predictionResults = {'hh_pred':pred_hh, 'qcd_pred':pred_qcd}
        sig, cut, sigErr = compareManyHistograms( predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'CNN Score', 0, 1, _nBins, _yMax = 5, _normed=True, savePlot=True, saveDir=self.topDir, writeSignificance=True, _testingFraction=self.testingFraction )
      
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
        self.predictions_test = self.best_model.predict(self.images_test)
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





