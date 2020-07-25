##  Author:  Ben Tannenwald
##  Date:    Jan 19, 2020
##  Purpose: Class to hold functions for testing Lorentz Boost Network approach for distinguishing between dihiggs signal and qcd background


# Import the needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.utils import normalize, to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.regularizers import l1, l2
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

import json
import multiprocessing as mp
from time import sleep

from lbn import LBN, LBNLayer

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)
#tf.random.set_seed(seed)

import sys, os
sys.path.insert(0, '/home/btannenw/Desktop/ML/dihiggsMLProject/')
from utils.commonFunctions import *

topDir = '/home/btannenw/Desktop/ML/dihiggsMLProject/lorentzBoostNetwork/'



class lorentzBoostAnalyzer:

    def __init__ (self, nJets = 8, signalData=None, backgroundData=None, modelName='LBA', testingFraction=0.3):
        self.signalData = signalData
        self.backgroundData = backgroundData
        self.nJets = nJets
        #self.trainData   = trainData[0]
        #self.trainLabels = trainData[1]
        #self.testData    = testData[0]
        #self.testLabels  = testData[1]
        self.modelName = modelName
        self.testingFraction = testingFraction
        
        self.trainVectorsByEvent = np.empty([1,4*self.nJets])
        self.trainLabelsByEvent  = np.empty([1,2])
        self.testVectorsByEvent  = np.empty([1,4*self.nJets])
        self.testLabelsByEvent   = np.empty([1,2])
        
        self.model = None
        self.history = None
        self.best_model = None
        self.hyperparameters = {'nLBNParticles':            5,
                                'nodesInFirstHiddenLayer':  80, 
                                'nodesInSecondHiddenLayer': 256, 
                                'hiddenActivation':        'relu', 
                                'outputActivation':        'sigmoid', 
                                'lossFunction':            'binary_crossentropy',

        }
        

    def makeJetVectors(self):
        """ make sets for training and testing"""

        # *** 1. Make sets of labels
        jetLabels = [x for x in range(1, self.nJets+1)]
        jetVariables = ['energy', 'px', 'py', 'pz']
        variables_jetVects = ['jet{0}_{1}'.format(iJetLabel, iJetVariable) for iJetLabel in jetLabels for iJetVariable in jetVariables]
        
        # *** 2. Split testing and training
        jetVects_data_train, jetVects_data_test, jetVects_labels_train, jetVects_labels_test = makeTestTrainSamplesWithUserVariables(self.signalData, self.backgroundData, variables_jetVects, self.testingFraction)

        # *** 3. Make proper LBN input vectors
        jetType = 'jet'
        self.trainVectorsByEvent = self.returnJetVectorInputsToLBN( jetVects_data_train, jetType, self.nJets)
        self.testVectorsByEvent  = self.returnJetVectorInputsToLBN( jetVects_data_test, jetType, self.nJets)

        self.trainLabelsByEvent = np.array([[0.,1.] if x ==0 else [1.,0.] for x in jetVects_labels_train.isSignal]).astype(np.float32)
        self.testLabelsByEvent  = np.array([[0.,1.] if x ==0 else [1.,0.] for x in jetVects_labels_test.isSignal]).astype(np.float32)

        print(np.shape(self.trainVectorsByEvent))

        
        ## # *** Make signal/background specific containers for labelled evaluation 
        ## hh_data_test, hh_labels_test, qcd_data_test, qcd_labels_test = returnTestSamplesSplitIntoSignalAndBackground(testVectorsByEvent, testLabelsByEvent)

    def returnJetVectorInputsToLBN( self, _df, _jetType='jet', _nJets=4):
    
        #flattened
        _allVectorsFlattened = None
        _var = [_jetType+'{}_energy', _jetType+'{}_px', _jetType+'{}_py', _jetType+'{}_pz']
    
        for i in range(1, _nJets + 1):
            _varN = [x.format(i) for x in _var]
            _jetNData = _df[ _varN ].astype(np.float32)
            _vectN = [list(x) for x in _jetNData.values]
        
            if _allVectorsFlattened == None:
                _allVectorsFlattened = _vectN
            else:
                _allVectorsFlattened = [ x + y for x,y in zip(_allVectorsFlattened, _vectN) ]

        return np.array(_allVectorsFlattened)
    

    # *** 3A. Define LBN model and train
    def createModelLBN( self, user_hyperparameters={}, _weightsDir=''):
        """make lbn model"""
    
        print("++ Setting hyperparameters...")
        for hp in self.hyperparameters.keys():
            if hp in user_hyperparameters.keys():
                self.hyperparameters[hp] = user_hyperparameters[hp] 
        
            print("{} = {}".format(hp, self.hyperparameters[hp]))
    
    
        #init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1, seed=123)
 
        features = ["E", "pt", "eta", "phi", "m", "pair_dr"]
        lbn_layer = LBNLayer(n_particles= self.hyperparameters['nLBNParticles'], boost_mode="pairs", features=features)
    
    
        metrics = [
            tf.keras.metrics.categorical_accuracy,
            tf.keras.metrics.AUC(name='auc'),
        ]
        
        l2_reg = tf.keras.regularizers.l2(1e-4)
    
        dense_kwargs_IML = dict(
            activation="selu",
            kernel_initializer=tf.keras.initializers.lecun_normal(),
            kernel_regularizer=l2_reg,
        )
        
        dense_kwargs = dict(
            activation = self.hyperparameters['hiddenActivation'],
            kernel_initializer=tf.keras.initializers.lecun_normal(),
            kernel_regularizer=l2_reg,
        )

        _model = tf.keras.models.Sequential()
        
        #_model.add(LBNLayer(5, boost_mode=LBN.PAIRS, features=features))
        _model.add(lbn_layer)
        _model.add(tf.keras.layers.BatchNormalization(axis=1))
        
        
        _model.add(tf.keras.layers.Dense( self.hyperparameters['nodesInFirstHiddenLayer'], **dense_kwargs))
        _model.add(tf.keras.layers.Dense( self.hyperparameters['nodesInSecondHiddenLayer'], **dense_kwargs))
        
        
        #self.model.add(tf.keras.layers.Dense(750, activation='relu'))#, kernel_regularizer=l2_reg))
        #self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        #self.model.add(tf.keras.layers.Dropout(0.2))
        
        #self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        #self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        #self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        
        _model.add(tf.keras.layers.Dense(2, activation= self.hyperparameters['outputActivation'], kernel_regularizer=l2_reg))
        
        _model.compile(loss= self.hyperparameters['lossFunction'],
                       optimizer='adam',
                       metrics = metrics
        )
        
        if _weightsDir !='':
            
            local_dir = os.path.join(topDir, "lbn", "models", _weightsDir)
            modelfile = os.path.join(local_dir, _weightsDir)+'.hdf5'
            print("++ loading model from {}".format(modelfile))
            #<-- FIXME: this does not check if file exits
            
            _model.predict(np.empty([1,32]))
            _model.load_weights(modelfile)

        
        return _model


    def fit_model( self, epochs=10, batch_size=512, model_hyperparams={}, patience=100, modelName='', verbose=1):
           
        # *** 0. Create model
        _model = self.createModelLBN(model_hyperparams)
        modelName = self.modelName if modelName == '' else modelName
                
        # *** 1. Define output directory
        model_dir = self.getModelDir( modelName )
            
        # *** 2. Define callbacks for training
        fit_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, modelName)+'.hdf5',
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
                patience=patience, 
                min_delta=.0025,
            ),
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(model_dir, modelName)+'_training.log',
                separator=';',
                append=False,
            ),
        ]
        
        # *** 3. Safety checks for data
        data = (self.trainVectorsByEvent, self.trainLabelsByEvent)
        validation_data = (self.testVectorsByEvent, self.testLabelsByEvent)

        
        # *** 4. Fit!!
        print('++ Begin model training\n')
        self.history = _model.fit(data[0], data[1],
                                  validation_data=validation_data,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  callbacks=fit_callbacks,
                                  verbose=verbose,
        )
        self.model = _model
    
        # *** 5. Save hyperparameters
        hp_filename=os.path.join(model_dir, modelName)+'_hyperparameters.json'
        with open(hp_filename, 'w') as outfile:
            json.dump( self.hyperparameters, outfile)


        # *** 6. Save history plots
        makeHistoryPlots( self.history, ['categorical_accuracy', 'loss', 'auc'], 'LBN', savePlot=True, saveDir=model_dir)

        # *** 7. Store best model
        self.load_model( modelName, best_model=True )

        #return self.model, self.history


    def fit_swarm( self, epochs=10, batch_size=512, model_hyperparams={}, patience=100):
        """ some fitting with particle swarm"""

        # *** 0. Figure out what step we're on ... I worry about this when running in parallel
        model_dir = self.getModelDir('')
        _modelStem = self.modelName + '_step'
        _previousRuns = []
        _currentProcess = mp.current_process()
        _swarmID = int(str(_currentProcess).split('-')[1].split(',')[0])

        print(_currentProcess, _currentProcess.pid, _swarmID)

        for d, s, f in os.walk(model_dir):
            for dd in d.split('\n'):
                if _modelStem in dd:
                    _previousRuns.append(int(dd.split(_modelStem)[1]))

        _nextRun = (max(_previousRuns) if len(_previousRuns)>0 else 0) + _swarmID
        
        # *** 1. Fit Xth iteration of model
        modelName = '{}_step{}'.format(self.modelName, _nextRun)
        self.fit_model(epochs=epochs, batch_size=batch_size, model_hyperparams=model_hyperparams, patience=patience, modelName = modelName, verbose=0)

        # *** 2. Evaluate best_model and return AUC?
        evals = self.best_model.evaluate(self.testVectorsByEvent, self.testLabelsByEvent, verbose=0) # reutnrs list of [loss, accuracy, auc]

        # ** 3. Make Test model
        sig, cut, err = self.test_model( self.best_model, modelName=modelName, savePlots=True )

        
        return (1-evals[2]) # FIXME... figure out how to get PSO to try to maximize instead of minimize


    def load_model(self, modelDirectory, best_model=False):
        """load previously created model"""
    
        # *** 0. Get hyperparameters
        local_dir = os.path.join(topDir, "lbn", "models", modelDirectory)
        hp_file = os.path.join(local_dir, modelDirectory)+'_hyperparameters.json'
        print("++ loading hyperparameters from {}".format(hp_file))
        
        with open(hp_file, 'r') as infile:
            self.hyperparameters = json.load(infile)
            
        # *** 1. Create model
        #self.model, self.hyperparameters = self.createModelLBN( self.hyperparameters, modelDirectory)
        _model = self.createModelLBN( self.hyperparameters, modelDirectory)

        if best_model == True:
            self.best_model = _model
        else:
            self.model = _model

        #return self.model
    

    def test_model( self, _model, modelName='', savePlots=False):
        """test model for significance etc"""

        # *** 0. Define output directory if appropriate
        modelName = self.modelName if modelName == '' else modelName
        model_dir = self.getModelDir( modelName ) + '/'        
        
        # *** 1. Split testing data into signal and background samples
        hh_data_test, hh_labels_test, qcd_data_test, qcd_labels_test = returnTestSamplesSplitIntoSignalAndBackground(self.testVectorsByEvent, self.testLabelsByEvent)

        # """ 2. Make predictions
        pred_hh = _model.predict(np.array(hh_data_test))
        pred_qcd = _model.predict(np.array(qcd_data_test))

        # *** 3. Make plot of prediction results
        print("++ Plotting test sample prediction results")
        _nBins = 40
        predictionResults = {'hh_pred':pred_hh[:,0], 'qcd_pred':pred_qcd[:,0]}
        compareManyHistograms( predictionResults, ['hh_pred', 'qcd_pred'], 2, 'Signal Prediction', 'LBN Signal Score', 0, 1, _nBins, _yMax = 5, _normed=True, savePlot=savePlots, saveDir=model_dir, writeSignificance=True, _testingFraction=self.testingFraction)

        # *** 4. Make ROC curve
        print("++ Making ROC curve")
        testPredsByEvent = _model.predict(self.testVectorsByEvent.copy())
        makeEfficiencyCurves( dict(label="LBN+DNN", labels=self.testLabelsByEvent, prediction=testPredsByEvent, color="blue"), saveDir=model_dir, _modelName=modelName, savePlot=savePlots)

        # *** 5. Get best cut value for model assuming some minimal amount of signal
        print("++ Calculating best significance")
        sig, cut, err = returnBestCutValue('ff-NN', pred_hh[:,0].copy(), pred_qcd[:,0].copy(), _minBackground=200, _testingFraction=self.testingFraction)

        return sig, cut, err


    def getModelDir(self, modelName):
        """ common function for setting model directory"""

        # *** 1. Define output directory and create if necessary
        model_dir = os.path.join(topDir, "lbn", "models", modelName)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return model_dir
    
    def runIterations( self, iterations=5, epochs=10, batch_size=512, model_hyperparams={}, patience=100 ):
        """ run user-specified iterations on same model"""

        results={}

        # *** 1. Iteration
        for step in range(0, iterations):

            # ** A. Fit Xth iteration of model
            modelName = '{}_step{}'.format(self.modelName, step)
            self.fit_model(epochs=epochs, batch_size=batch_size, model_hyperparams=model_hyperparams, patience=patience, modelName = modelName)
                        
            # ** B. Test model
            sig, cut, err = self.test_model( self.best_model, modelName=modelName, savePlots=True )
            
            # ** C. Store best cut value in dictionary
            results[step] = {'Sig': sig, 'Error': err, 'Cut': cut}

        print (results)
        
        # *** 2. Find best
        best_iteration = -1
        for iStep in results:
            if best_iteration == -1:
                best_iteration = iStep
            else:
                if results[iStep]['Sig'] > results[best_iteration]['Sig']:
                    best_iteration = iStep

        print("\nBest iteration was step {}:".format(best_iteration))
        print('Significance = {} +/- {} with score > {}\n'.format( round(results[best_iteration]['Sig'], 3), round(results[best_iteration]['Error'], 3), round(results[best_iteration]['Cut'], 3)))


        

