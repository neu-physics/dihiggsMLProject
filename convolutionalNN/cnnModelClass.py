#!/usr/bin/env python
##  Author:  Ben Tannenwald
##  Date:    April 16, 2020
##  Purpose: Class to load data, create user-defined CNN model, train CNN, evaluate performance, and save the trained model


# To-Do
#1) transfer files locally from lxplus or maybe figure out venv on lxplus
#X) CM of phi for image?
#3) bb-center a la photography talk
#4) slide 9: remake these for diHiggs events that have the Higgs back-to-back ie no ISR. Are the blobs more pronounced than non-back-to-back diHiggs events?
#X) increase pixels to see if better? --> 15 -> 25 not really improved
#6) compare expected yields post-selection from this to other approaches. Much higher signal stats here! But by how much?
#7) subtract average a la Anna's idea
#X) store images by jet category?
# ...
# ...
#X) script image making --> different pixel sizes, jet categories
#10) compare ROC curves for different approaches a la Evan's work
#11) make cnn training class-script for automation -> saves models, roc curve as member, etc



# Import the needed libraries
import os
import tensorflow as tf
import numpy as np
import matplotlib
if "_CONDOR_SCRATCH_DIR" in os.environ:
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
from utils.commonFunctions import *



class cnnModelClass:
    
    def __init__ (self, _modelName, _cnnLayers, _ffnnLayers, _hhFile, _qcdFile, _imageCollection, _loadSavedModel = False, _testRun = False):
        self.modelName   = _modelName
        self.cnnLayers = _cnnLayers
        self.ffnnLayers = _ffnnLayers
        self.hhFile = _hhFile
        self.qcdFile = _qcdFile
        self.imageCollection = _imageCollection
        self.isTestRun = _testRun
        self.loadSavedModel = _loadSavedModel

        # Class Defaults
        #self.transparency = 0.5  # transparency of plots

        # Global Variables 
        self.hh = []
        self.qcd = []
        self.model = []
        self.model_history = []
        self.images_train = []
        self.labels_train = []
        self.images_test  = []
        self.labels_test  = []
        #self.outputDataForLearning = []

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

        return

    ##############################################################
    ##                        functions                         ##
    ##############################################################

    
    def processInputs(self):
        """ open files and load images"""

        # Load files
        hh_h5  = h5.File( self.hhFile, 'r')
        qcd_h5  = h5.File( self.qcdFile , 'r')

        # Print image collection names
        print("+++ Available image collections: {}".format( ",".join(hh_h5.keys()) ))
        
        # Load user-specified image collection
        if self.isTestRun:
            _nEventsForTesting = 1000
            self.hh  = hh_h5[ self.imageCollection ][:_nEventsForTesting]
            self.qcd = qcd_h5[ self.imageCollection ][:_nEventsForTesting]
        else:
            self.hh = hh_h5[ self.imageCollection ]
            self.qcd = qcd_h5[ self.imageCollection]
            
        print("+ {} hh images, {} qcd images".format( len(self.hh), len(self.qcd)))

        # Make labels
        hh_labels = np.ones( len(self.hh) )
        qcd_labels = np.zeros( len(self.qcd) )
        
        # Make combined dataset
        all_images = np.concatenate ( (self.hh, self.qcd) )
        all_labels = np.concatenate ( (hh_labels.copy(), qcd_labels.copy()), axis=0)
        print("+ images shape: {}, labels shape:{}".format( all_images.shape, all_labels.shape) )

        # Create training and testing sets
        self.images_train, self.images_test, self.labels_train, self.labels_test = train_test_split(all_images, all_labels, test_size=0.25, shuffle= True, random_state=30)

        
        return

    
    def singleEventImages(self, isSignal = False):
        """ make a few single event images"""

        if isSignal:
            imgs = self.hh
        else:
            imgs = self.qcd
            
        evt = 503
        plt.imshow(imgs[evt], alpha=0.88)
        plt.show()
        plt.imshow(imgs[evt+1], alpha=0.88)
        plt.show()
        plt.imshow(imgs[evt+2], alpha=0.88)
        plt.show()
        plt.imshow(imgs[evt+3], alpha=0.88)
        plt.clim(.5, 1)
        plt.show()

        return

    
    def makeCNN(self):
        """ create CNN and return compiled model"""
        print("+++ Make CNN")

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
        total = len(self.qcd) + len(self.hh)
        weight_for_0 = (1 / len(self.qcd))*(total)/2.0 
        weight_for_1 = (1 / len(self.hh))*(total)/2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print('+ Weight for class 0: {:.2f}'.format(weight_for_0))
        print('+ Weight for class 1: {:.2f}'.format(weight_for_1))
        

        # *** 1. Define model
        self.model = Sequential()
        pixelWidth = self.images_train.shape[1]
        
        # ** A. Convolutional component
        firstLayer = True
        for layer in self.cnnLayers:
            # layer format example: [ "Conv2D", [16, (3,3)]] or ["MaxPooling2D", [(3, 3)]]
            if layer[0] == "Conv2D" and firstLayer:
                self.model.add( Conv2D( layer[1][0], layer[1][1], input_shape=(pixelWidth, pixelWidth, 3), **conv_kwargs) )
                firstLayer = False
            elif layer[0] == "Conv2D" and not firstLayer:
                self.model.add( Conv2D( layer[1][0], layer[1][1], **conv_kwargs) )
            elif layer[0] == "MaxPooling2D":
                self.model.add( MaxPooling2D( layer[1][0]) )

        # ** B. Flatten model for input to feed-forward network
        self.model.add( Flatten() ) 

        # ** C. Feed-forward component
        for layer in self.ffnnLayers:
            # layer format example: [ "Dense", [64]] or ["BatchNormalization"], ["Dropout", [0.2]]
            if layer[0] == "Dense":
                self.model.add( Dense( layer[1][0], **dense_kwargs) )
            elif layer[0] == "BatchNormalization":
                self.model.add(  BatchNormalization() )
            elif layer[0] == "Dropout":
                self.model.add( Dropout( layer[1][0]) )

        
        # ** D. Output layer
        self.model.add( Dense(1, activation='sigmoid', bias_initializer=output_bias) )
    
        # ** E. Print summary
        print("++ Model Summary\n {}".format(self.model.summary()))

        # ** F. Define metrics and compile
        metrics = [ tf.keras.metrics.categorical_accuracy,
                    'accuracy',
                    tf.keras.metrics.AUC(name='auc'),
        ]

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=metrics,
        )

        # **G. Load model if appropriate
        if self.loadSavedModel:
            local_dir = os.path.join(self.modelName, "models")
            modelfile = os.path.join(local_dir, self.modelName)+'.hdf5'
            print("++ loading model from {}".format(modelfile))
            if not os.path.isfile(modelfile):
                print("--- ERROR, Modelfile {} NOT FOUND. No weights initialized".format(modefile))
                return
        
            self.model.predict( np.empty( (1, pixelWidth, pixelWidth, 3) ))
            self.model.load_weights(modelfile)

            
        return


    def trainCNN(self):
        print("+++ Train CNN" )

        # *** 1. Define output directory
        topDir = self.modelName
        name = self.modelName
        if not os.path.exists(topDir):
            os.makedirs(topDir)
            model_dir = os.path.join(topDir, "", "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        model_dir = os.path.join(topDir, "", "models")
                
        # *** 2. Define callbacks for training
        fit_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, name)+'.hdf5',
                save_best_only=True,
                save_weights_only=True,
                #monitor="val_auc",
                #mode="max",
                monitor="val_loss",
                mode="min",
            ),
            tf.keras.callbacks.EarlyStopping(
                #monitor="val_auc",
                #mode="max",
                monitor='val_loss', 
                mode='min', 
                verbose=1, 
                patience=15,  
                min_delta=.0025,
            ),
        ]
        
        
        # *** 3. Train model
        nEpochs = 10 if self.isTestRun else 50
        self.model_history = self.model.fit(self.images_train, self.labels_train, epochs=nEpochs, 
                                            shuffle=True,
                                            batch_size=512,
                                            validation_data=(self.images_test, self.labels_test),
                                            callbacks=fit_callbacks,
                                            #class_weight=class_weight
        )
        
        return
    


    def evaluateModel(self):
        """ evaluateModel( modelNoWeights, historyNoWeights, self.images_test, self.labels_test) """
        print("+++ Evaluate Model")
        
        # *** 0. set inputs --> special treatment if loading saved model
        images = []
        labels = []
        if self.loadSavedModel:
            images = np.concatenate ( (self.images_test.copy(), self.images_train.copy()), axis=0)
            labels = np.concatenate ( (self.labels_test.copy(), self.labels_train.copy()), axis=0)
            images, labels = shuffle(images, labels, random_state=0)
        else:
            images = self.images_test
            labels = self.labels_test
        
        # *** 1. Make prediction data
        hh_data_test    = np.asarray([x for x,y in zip( images, labels) if y==1])
        hh_labels_test  = np.asarray([y for x,y in zip( images, labels) if y==1])
        qcd_data_test   = np.asarray([x for x,y in zip( images, labels) if y==0])
        qcd_labels_test = np.asarray([y for x,y in zip( images, labels) if y==0])
                
        # *** 2. Make history plots
        if not self.loadSavedModel:
            print("++ Training History Plots")
            makeHistoryPlots( self.model_history, ['accuracy', 'loss', 'auc'], self.modelName, savePlot=True, saveDir=self.modelName )
        
        # *** 3. Make predictions
        score_hh = self.model.evaluate(hh_data_test, hh_labels_test)
        score_qcd = self.model.evaluate(qcd_data_test, qcd_labels_test)
        print(score_hh, score_qcd)
        pred_hh = self.model.predict(hh_data_test)
        pred_qcd = self.model.predict(qcd_data_test)
        
        # *** 4. make output score plot
        print("++ Output Score Plot")
        _nBins = 40
        predictionResults = {'hh_pred':pred_hh, 'qcd_pred':pred_qcd}
        compareManyHistograms( predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'CNN Score', 0, 1, _nBins, _yMax = 5, _normed=True, savePlot=True, saveDir=self.modelName )
        
        # *** 5. Get best cut value for CNN assuming some minimal amount of signal
        pred_hh_sig = [x[0] for x in pred_hh.copy()]
        pred_qcd_sig = [x[0] for x in pred_qcd.copy()]
        
        sig, cut, sigErr = returnBestCutValue('CNN', pred_hh_sig, pred_qcd_sig, _minBackground=400e3, _testingFraction=0.025)

        ## *** 5B. Get signifiance for any user-specified NN score cut value
        #testingFraction = 0.1
        #lumiscale_hh  = getLumiScaleFactor(testingFraction, True, 25e3)
        #lumiscale_qcd = getLumiScaleFactor(testingFraction, False, 50e3)
        #cut = 0.48
        #_nSignal = sum( value > cut for value in pred_hh_sig)*lumiscale_hh
        #_nBackground = sum( value > cut for value in pred_qcd_sig)*lumiscale_qcd
        #
        #print('nSig = {0} , nBkg = {1} with significance = {2} for NN score > {3}'.format(_nSignal, _nBackground, _nSignal/np.sqrt(_nBackground), cut) )

        # *** 6. Confusion matrix
        print("++ Confusion Matrix")
        preds_test = self.model.predict(self.images_test)
        cm = confusion_matrix( self.labels_test, preds_test > cut)
        print(cm)
        print('QCD called QCD (True Negatives): {} ({}%)'.format( cm[0][0], round(100*(cm[0][0]/sum(cm[0]))) ))
        print('QCD called Dihiggs (False Positives):  {} ({}%)'.format( cm[0][1], round(100*(cm[0][1]/sum(cm[0]))) ))
        print('Dihiggs called QCD (False Negatives):  {} ({}%)'.format( cm[1][0], round(100*(cm[1][0]/sum(cm[1]))) ))
        print('Dihiggs called Dihiggs (True Positives):  {} ({}%)'.format( cm[1][1], round(100*(cm[1][1]/sum(cm[1]))) ))
        print('Total Dihiggs: ', np.sum(cm[1]))
        
        # *** 7. Make ROC curve
        print("++ ROC Curve")
        makeEfficiencyCurves( dict(label="CNN", labels=self.labels_test, prediction=preds_test, color="blue"), _modelName = self.modelName, savePlot=True, saveDir=self.modelName)
        #utils/commonFunctions.py: def makeEfficiencyCurves(*data, _modelName='', savePlot=False, saveDir=''):

        return


#==========================================================================================



## *** 7. Make overlaid ROC curves
#predsNoWeights = modelNoWeights.predict(self.images_test)
#predsWeighted  = modelWeighted.predict(self.images_test)
#overlay = [ dict(label="No Weights", labels=self.labels_test, prediction=predsNoWeights, color="blue"),
#           dict(label="Weighted", labels=self.labels_test, prediction=predsWeighted, color="red")
#         ]
#overlayROCCurves( overlay )





