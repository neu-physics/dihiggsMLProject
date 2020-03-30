##  Author:  Ben Tannenwald
##  Date:    April 1, 2019
##  Purpose: Class to hold functions for testing higgs reconstruction algorithms and outputting .csv file for ML training

import uproot, uproot_methods
import uproot_methods.classes.TLorentzVector as TLorentzVector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy, csv, os, itertools
from JetCons import JetCons
import pickle


class eventReconstruction:
    
    def __init__ (self, _datasetName, _inputFile, _isDihiggsMC, _isTestRun = False, _saveIdx = 0):
        self.datasetName   = _datasetName
        self.inputFileName = _inputFile
        self.isTestRun     = _isTestRun
        self.isDihiggsMC   = _isDihiggsMC
        self.saveIdx       = _saveIdx
        if os.path.isdir( self.datasetName )==False:
            os.mkdir( self.datasetName )

        # Class Defaults
        self.transparency = 0.5  # transparency of plots
        self.dR_cut_quarkToJet = 0.40
        self.mass_higgs = 125.0 #GeV
        self.width_higgs = 15.0 #GeV, reco width
        self.minJetPt = 20.0 #GeV
        self.maxJetAbsEta = 2.5
        self.nJetsToStore = 4
        self.requireTags = True
        self.requireNTags = 4
        self.ptOrdered = True
        self.considerFirstNjetsInPT = -1
        self.saveAlgorithm = 'equalDijetMass'
        self.saveLowLevelVariablesForTraining = True
        self.saveJetConstituents = False

        # Global Variables 
        self.outputDataForLearning = []
        self.pairingAlgorithms = ['minHarmonicMeanDeltaR', 'closestDijetMassesToHiggs', 'equalDijetMass', 'equalDeltaR', 'dijetMasses']
        self.variableCategoryDict = {'All':[], 'Matchable':[], 'Best':[], 'Best+Matchable':[], 'Correct':[]}
        
        # Jet constituents information
        self.jetConsCand_Names = ["EFlowTrack", "EFlowNeutralHadron", "EFlowPhoton"]
        self.jetConsCand_Keys = ["fUniqueID", "PT", "Eta", "Phi","P"]
        self.jetConsOutputKeys = ["UID", "PT", "Eta", "Phi", "E", "rapidity", "type"]
        self.outputJetConsInfo = []

        self.cutflowDict = { 'All':0, 'Matchable':0, 'Fully Matched':0, '>= 1 Pair Matched':0}
        self.jetTagCategories = ['Incl',
                                 '0jIncl', '0j0b',  
                                 '1jIncl', '1j0b', '1j1b',  
                                 '2jIncl', '2j0b', '2j1b', '2j2b', 
                                 '3jIncl', '3j0b', '3j1b', '3j2b', '3j3b', 
                                 '4jIncl', '4j0b', '4j1b', '4j2b', '4j3b', '4j4b', 
                                 '5jIncl', '5j0b', '5j1b', '5j2b', '5j3b', '5j4b', '5j5b', 
                                 '6jIncl', '6j0b', '6j1b', '6j2b', '6j3b', '6j4b', '6j5b', '6j6b',
                                 '7jIncl', '7j0b', '7j1b', '7j2b', '7j3b', '7j4b', '7j5b', '7j6b', '7j7b']

        self.plottingData = {algorithm:copy.deepcopy(self.variableCategoryDict) for algorithm in self.pairingAlgorithms}
        self.eventCounterDict = { algorithm:{category:copy.deepcopy(self.cutflowDict) for category in self.jetTagCategories} for algorithm in self.pairingAlgorithms}
        self.nBTagsPerEvent  = []
        self.nJetsPerEvent   = []

        # Per-Event Variables
        self.thisEventIsMatchable = False
        self.thisEventWasCorrectlyMatched = False
        self.nJets  = 0
        self.nBTags = 0
        self.quarkIndices  = []
        self.jetIndices    = []
        self.allJetIndices = []
        self.matchedQuarksToJets = {}
        self.jetVectorDict = {}
        self.quarkVectorDict = {}
        self.jetCutflow = {'all':0, 'pt+eta':0, 'tagged':0, 'added':0}
        self.jetCounter = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0}
        self.matchedCounter = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0}

        # Branch Definitions
        self.delphesFile      = uproot.rootio.TObject
        self.inputDF          = None
        self.nEntries         = -1
        self.l_genPID         = []
        self.l_genD1          = []
        self.l_genD2          = []
        self.l_genStatus      = []
        self.l_genPt          = []      
        self.l_genEta         = []
        self.l_genPhi         = []
        self.l_genMass        = []
        self.l_jetPt          = []
        self.l_jetEta         = []
        self.l_jetPhi         = []
        self.l_jetMass        = []
        self.l_jetBTag        = []
        self.l_jetCons        = []
        self.l_missingET_met  = []
        self.l_missingET_phi  = []
        self.l_scalarHT       = []
        self.jetConsCand      = []      #jet constituent candidate


    ##############################################################
    ##                           MAIN                           ##
    ##############################################################

    def runReconstruction(self):

        self.printAllOptions()
        self.initFileAndBranches()
        self.outputVariableNames = self.createOutputVariableList()


        for iEvt in range(0,self.nEntries):
            # *** 0. Kick-out condition for testing
            if iEvt > 40 and self.isTestRun is True:
                break
            if iEvt%2000==0:
                print("Analyzing event number",iEvt)


            # *** 1. Get truth information
            self.getTruthInformation( iEvt )

            # *** 2. Get jet reco information
            self.getRecoInformation( iEvt )
            #if (self.requireTags==True and self.nBTags < self.requireNTags) or (self.requireTags==False and self.nJets < 4): continue 
            if (self.requireTags==True and ( (self.nBTags < self.requireNTags) or len(self.jetIndices) < 4 )) or (self.requireTags==False and self.nJets < 4): continue 

            # *** 3. Do some quark-to-jet truth matching
            self.truthToRecoMatching( iEvt )

            # *** 4. Evaluate all pairing algorithms
            self.evaluatePairingAlgorithms( iEvt )


        # *** 5. Store output data in .csv for later usage
        self.writeDataForTraining()

        # *** 6. Store output jet constituent data in .pkl
        if(self.saveJetConstituents==True):
            self.writeJetConsData()

        print( "Finished processing {0} events...".format(self.nEntries) )

        print('jetCounter', self.jetCounter)
        print('matchedCounter', self.matchedCounter)
        print('jetCutflow', self.jetCutflow)

        return


    def printAllAlgorithmEventTotals(self, _jetTagCategories):

        # *** 1. Print event counter info 
        for algorithm in self.pairingAlgorithms:
            for jetTagCat in _jetTagCategories:
                self.printEventCounterInfo(algorithm, jetTagCat)
    
        return


    def makeAlgorithmComparisonPlots(self):

        # *** 1. Save general plots
        self.plotOneHistogram(self.nJetsPerEvent, 0, 'Number of Jets Per Event', 'Number of Jets', 0, 10, 11 )
        self.plotOneHistogram(self.nBTagsPerEvent, 1, 'Number of b-Tagged Jets Per Event', 'Number of b-Tags', 0, 10, 11 )

        # *** 2. Save algorithm-specific plots
        for pairingAlgorithm in self.pairingAlgorithms:
            # ** A. Get plotting options, e.g. n_bins, xmin/xmax, title, etc
            _plotOpts = self.returnPlottingOptions( pairingAlgorithm )

            # ** B. save plots with truth-matching information for dihiggs MC
            if self.isDihiggsMC:
                self.compareManyHistograms( pairingAlgorithm, ['All', 'Matchable'], 2, 'All Jet Pairs ' + _plotOpts[0], 'Jet Pair ' + _plotOpts[1], int(_plotOpts[2]), int(_plotOpts[3]), int(_plotOpts[4]) )
                self.compareManyHistograms( pairingAlgorithm, ['Best', 'Best+Matchable', 'Correct'], 3, 'Best Jet Pair ' + _plotOpts[0], 'Jet Pair ' + _plotOpts[1], int(_plotOpts[2]), int(_plotOpts[3]), int(_plotOpts[4]) )
                self.compareManyHistograms( pairingAlgorithm, ['All', 'Matchable', 'Correct', 'Best', 'Best+Matchable'], 5, 'Best Jet Pair ' + _plotOpts[0], 'Jet Pair ' + _plotOpts[1], int(_plotOpts[2]), int(_plotOpts[3]), int(_plotOpts[4]), _normed=True)
            # ** C. save plots without if QCD
            else:
                self.compareManyHistograms( pairingAlgorithm, ['All', 'Best'], 2, 'All Jet Pairs ' + _plotOpts[0], 'Jet Pair ' + _plotOpts[1], int(_plotOpts[2]), int(_plotOpts[3]), int(_plotOpts[4]) )
                self.compareManyHistograms( pairingAlgorithm, ['All', 'Best'], 5, 'Best Jet Pair ' + _plotOpts[0], 'Jet Pair ' + _plotOpts[1], int(_plotOpts[2]), int(_plotOpts[3]), int(_plotOpts[4]), _normed=True)        
        
        return

    def makeEfficiencyPlots(self):

        _fullyMatchedDict, _onePairMatchedDict, _labels = self.listOfEfficienciesForAlgorithm()

        self.calculateAndDrawEfficiency( _fullyMatchedDict, _labels, 'fullyMatched' )
        #self.calculateAndDrawEfficiency( _onePairMatchedDict, _labels, 'onePairMatched' )

        return

    ##############################################################
    ##             FUNCTIONS TO SET/GET VARIABLES               ##
    ##############################################################

    def setTransparency(self, _userTransparency):
        self.transparency = _userTransparency
    def getTransparency(self):
        print ("Transparency: ", self.transparency)

    def setNJetsToStore(self, _userNJetsToStore):
        self.nJetsToStore = _userNJetsToStore
    def getNJetsToStore(self):
        print ("N_Jets To Store: ", self.nJetsToStore)
        
    def setQuarkToJetCutDR(self, _userQuarkToJetCutDR):
        self.dR_cut_quarkToJet = _userQuarkToJetCutDR
    def getQuarkToJetCutDR(self):
        print ("DeltaR Cut Between Jet and Quark: ", self.dR_cut_quarkToJet)

    def setHiggsMass(self, _userMassHiggs):
        self.mass_higgs = _userMassHiggs
    def getHiggsMass(self):
        print ("Higgs Mass: ", self.mass_higgs)

    def setHiggsWidth(self, _userWidthHiggs):
        self.width_higgs = _userWidthHiggs
    def getHiggsWidth(self):
        print ("Higgs Width: ", self.width_higgs)

    def setJetPtEtaCuts(self, _userJetPtCut, _userJetEtaCut):
        self.minJetPt = _userJetPtCut #GeV
        self.maxJetAbsEta = _userJetEtaCut
    def getJetPtEtaCuts(self):
        print ("Minimum Jet pT [GeV]: ", self.minJetPt)
        print ("Maximum Jet |eta|: ", self.maxJetAbsEta)

    def setSaveAlgorithm(self, _userSaveAlgorithm):
        if _userSaveAlgorithm not in self.pairingAlgorithms:
            print ("!!! WARNING, {0} not a valid algorithm for saving. Returning to default: {0} . Please try again.".format(_userSaveAlgorithm, self.saveAlgorithm))
        else:
            self.saveAlgorithm = _userSaveAlgorithm
    def getSaveAlgorithm(self):
        print ("Algorithm saved for training: ", self.saveAlgorithm)

    def setSaveLowLevelVariables(self, _userSaveLowLevel):
        self.saveLowLevelVariablesForTraining = bool(_userSaveLowLevel)
    def getSaveLowLevelVariables(self):
        print ("Save low-level variables for training: ", self.saveLowLevelVariablesForTraining)

    def setConsiderFirstNjetsInPT(self, _userFirstNjets):
        self.considerFirstNjetsInPT = int(_userFirstNjets)
    def getConsiderFirstNjetsInPT(self):
        print ("Consider first N jets in pT, N = ", self.considerFirstNjetsInPT)

    def setRequireTags(self, _requireTags):
        self.requireTags = bool(_requireTags)
    def getRequireTags(self):
        print ("Require only b-tagged jets: ", self.requireTags)

    def setRequireNTags(self, _requireNTags):
        self.requireNTags = int(_requireNTags)
    def getRequireNTags(self):
        print ("Require at least N b-tagged jets: N = ", self.requireNTags)

    def setPtOrdered(self, _ptOrdered):
        self.ptOrdered = bool(_ptOrdered)
    def getPtOrdered(self):
        print ("Order jets by pT: ", self.ptOrdered)

    def setSaveJetConstituents(self, _saveJetCons):
        self.saveJetConstituents = bool(_saveJetCons)
    def getSaveJetConstituents(self):
        print ("Save Jet Constituents: ", self.saveJetConstituents)


    def printAllOptions(self):
        print("=========   Options for {0}  =========".format(self.datasetName))
        self.getTransparency()
        self.getQuarkToJetCutDR()
        self.getHiggsMass()
        self.getHiggsWidth()
        self.getJetPtEtaCuts()
        self.getSaveAlgorithm()
        self.getHiggsWidth()
        self.getRequireTags()
        self.getPtOrdered()
        self.getConsiderFirstNjetsInPT()
        self.getNJetsToStore()
        print("======================================".format(self.datasetName))


    ##############################################################
    ##                FUNCTIONS FOR PLOTTING                    ##
    ##############################################################
    def returnPlottingOptions(self, _pairingAlgorithm):
        _plotOpts = []

        if _pairingAlgorithm == 'minHarmonicMeanDeltaR':
            _plotOpts = ['Delta R', 'Delta R', 0, 5.0, 100]
        elif _pairingAlgorithm == 'closestDijetMassesToHiggs':
            _plotOpts = ['Higgs Mass Diff', 'Higgs Mass Diff', 0, 50.0, 50]
        elif _pairingAlgorithm == 'equalDijetMass':
            _plotOpts = ['Dijet Mass Diff', 'Abs Dijet Mass Diff [GeV]', -0.0, 300.0, 100]
        elif _pairingAlgorithm == 'equalDeltaR':
            _plotOpts = ['Delta R', 'Delta R(h1, h2)', 0, 5.0, 100]
        elif _pairingAlgorithm == 'dijetMasses':
            _plotOpts = ['Dijet Mass', 'Dijet Mass [GeV]', 0, 600.0, 100]

        return _plotOpts

    def plotOneHistogram(self, _arr, _nPlot, _title, _xtitle, _xMin, _xMax, _nBins, _normed=False):
        #mean_arr = np.mean(arr)
        #stdev_arr = np.std(arr)
        #nEntries_arr = len(arr)
        
        #s1 = "Higgs Mass Reconstructed from 4 b-tagged jets:\n" \
            #     "entries = {}, mean = {:.4F}, std dev = {:.4F}".format(nEntries_arr, mean_arr, stdev_arr)
        
        plt.figure(_nPlot)
        plt.title(_title)
        plt.xlabel(_xtitle)
        _bins = np.linspace(_xMin, _xMax, _nBins)
        plt.hist(_arr, _bins, alpha=self.transparency, normed=_normed)
        #plt.legend(loc='upper right')
        #plt.text(10, 10, s1)

        # store figure copy for later saving
        fig = plt.gcf()
    
        # draw interactively
        plt.show()
    
        # save an image files
        _scope    = _title.split(' ')[0].lower()
        _variable = _xtitle.lstrip('Jet Pair').replace(' ','').replace('[GeV]','')
        _filename  = self.datasetName + '/' + _scope + '_' + _variable
        if _normed:
            _filename = _filename + '_norm'
        fig.savefig( _filename+'.png' )


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
        plt.figtext(.55, .5, _pairingAlgorithm)
    
        # store figure copy for later saving
        fig = plt.gcf()
    
        # draw interactively
        plt.show()
    
        # save an image files
        _scope    = _title.split(' ')[0].lower()
        _variable = _xtitle.lstrip('Jet Pair').replace(' ','').replace('[GeV]','')
        _allLabels = ''.join(_labels)
        _filename  = self.datasetName + '/' + _scope + '_' + _pairingAlgorithm + '_' + _allLabels + '_' + _variable
        if _normed:
            _filename = _filename + '_norm'
        fig.savefig( _filename+'.png' )

        return


    ##############################################################
    ##                FUNCTIONS FOR INDEXING                    ##
    ##############################################################

    def returnListOfTruthBQuarkIndicesByDaughters(self, _iEvent):
        ##################
        ### DEPRECATED ###
        ##################
        
        for iParticle in range(0, len(_D1)):
            if self.PID[_iEvent][iParticle]==25:
                _daughter1 = self.l_genD1[_iEvent][iParticle]
                _daughter2 = self.l_genD2[_iEvent][iParticle]
                _daughter1_PID = self.l_genPID[_iEvent][daughter1]
                _daughter2_PID = self.l_genPID[_iEvent][daughter2]
                #print('Event ',iEvt,'has higgs at position',iParticle,'with daughter1 (',daughter1,
                #    ') of PID',daughter1_PID,'and daughter2 (',daughter2,') of PID',daughter2_PID)
                if abs(_daughter1_PID) == 5 and abs(_daughter2_PID)==5:
                    self.quarkIndices.append(_daughter1)
                    self.quarkIndices.append(_daughter2)
    
        return 


    def returnListOfTruthBQuarkIndicesByStatus(self, _iEvent ):

        for iParticle in range(0, len(self.l_genStatus[_iEvent]) ):
            if self.l_genStatus[_iEvent][iParticle]==23:
                self.quarkIndices.append(iParticle)

        return 


    def addJetToList(self, _orderedJetIndices, _iEvent, _iJet):
        """ add jet to list of indices according to pt ordering (or not) depending on user specifications """

        if self.ptOrdered:
            _added = False
            #for index in range(0, len(_orderedJetIndices)):
            for index in _orderedJetIndices:
                if self.l_jetPt[_iEvent][_iJet] > self.l_jetPt[_iEvent][index] and _added==False:
                    _orderedJetIndices.insert(index, _iJet)
                    _added = True
                    self.jetCutflow['added'] += 1
                    
            if _added == False:
                _orderedJetIndices.append(_iJet)
                self.jetCutflow['added'] += 1
        else:
            _orderedJetIndices.append(_iJet)
        
        return _orderedJetIndices

    
    def returnNumberAndListOfJetIndicesPassingCuts(self, _iEvent):
        self.nJets = 0
        self.nBTags = 0
        self.jetIndices = []
        self.allJetIndices = []
        untaggedJetIndices = []
        taggedJetIndices   = []

        for iJet in range(0, len(self.l_jetPt[_iEvent])): 
            self.jetCutflow['all'] += 1

            if self.l_jetPt[_iEvent][iJet] > self.minJetPt and abs(self.l_jetEta[_iEvent][iJet]) < self.maxJetAbsEta and self.l_jetMass[_iEvent][iJet]>0: 
                # surpringly some jets (<1%) have negative mass. filter these out
                self.jetCutflow['pt+eta'] += 1
                self.nJets += 1

                if self.l_jetBTag[_iEvent][iJet] == 0: # un-tagged jets
                    untaggedJetIndices = self.addJetToList( untaggedJetIndices, _iEvent, iJet)
               
                elif self.l_jetBTag[_iEvent][iJet] == 1: # b-tagged jets
                    self.jetCutflow['tagged'] += 1
                    self.nBTags += 1
                    taggedJetIndices = self.addJetToList( taggedJetIndices, _iEvent, iJet)

        # ** set indices for consideration depending on user specificiations
        if self.requireTags:
             if self.considerFirstNjetsInPT==-1:
                 self.jetIndices.extend(taggedJetIndices)
                 self.jetIndices.extend(untaggedJetIndices)
             else: # need to take a particular number of jets
                 if len(taggedJetIndices) >= self.considerFirstNjetsInPT:
                     self.jetIndices.extend( taggedJetIndices[:self.considerFirstNjetsInPT] )
                 else:
                     self.jetIndices.extend( taggedJetIndices ) # add all b-tagged jets when N_tags < N_to-consider
                     addNUntaggedJets = self.considerFirstNjetsInPT - len(self.jetIndices)
                     self.jetIndices.extend( untaggedJetIndices[:addNUntaggedJets] ) # then add enough un-tagged jets to make up the difference
        else: # no b-tagging requirements, so only by ordering of jetIndex lists
            if self.considerFirstNjetsInPT==-1:
                self.jetIndices.extend(taggedJetIndices)
                self.jetIndices.extend(untaggedJetIndices)
            else: # take via pt orderding. WARNING [FIXME maybe] that this logic will break down if not picking jets by pt-ordering.. it's kind of hard-coded into the algorithm here
                i_taggedJet = 0
                i_untaggedJet = 0
                while len(self.jetIndices) < self.considerFirstNjetsInPT:
                    i_taggedPt = self.l_jetPt[ taggedJetIndices[ i_taggedJet ] ]
                    i_untaggedPt = self.l_jetPt[ untaggedJetIndices[ i_untaggedJet ] ]
                    
                    if i_taggedPt > i_untaggedPt:
                        self.jetIndices.append(i_taggedJet)
                        i_taggedJet += 1
                    elif i_untaggedPt > i_taggedPt:
                        self.jetIndices.append(i_untaggedJet)
                        i_untaggedJet += 1


        # add block to make list of jet indices up to nJetsToStore which can be greater than considerFirstNjetsInPT. these are jets for low-level analysis and not for use in higgs reconstruction algorithm
        self.allJetIndices = self.jetIndices.copy()
        i_jetParser = 0
        while (len(self.allJetIndices) < self.nJetsToStore):
            # run loop until either a) allJetIndices is length of nJetsToStore or break if exhausted event jetPt vector 
            if i_jetParser > len(self.l_jetPt[_iEvent])-1:
                break

            # append jet index if not already in list of indices
            if i_jetParser not in self.allJetIndices:
                self.allJetIndices.append(i_jetParser)

            # bump up iterator
            i_jetParser += 1

        # trim allJetIndices if necessary but this should never happen unless user has specified some weird shit
        self.allJetIndices = self.allJetIndices[:self.nJetsToStore]
        
        #print (self.nJets, self.nBTags, len(self.jetIndices), self.jetIndices, [self.l_jetPt[_iEvent][g] for g in self.jetIndices])
        
        return 


    def getDictOfQuarksMatchedToJets(self, _iEvent ): 
    
        for iQuark in self.quarkIndices:
            _tlv_quark = TLorentzVector.PtEtaPhiMassLorentzVector( self.l_genPt[_iEvent][iQuark], self.l_genEta[_iEvent][iQuark], self.l_genPhi[_iEvent][iQuark], self.l_genMass[_iEvent][iQuark]) 
            if iQuark not in self.quarkVectorDict.keys():
                self.quarkVectorDict[iQuark] = _tlv_quark
            
            for iJet in self.jetIndices:           
                # make vector for jets
                _tlv_jet = TLorentzVector.PtEtaPhiMassLorentzVector( self.l_jetPt[_iEvent][iJet], self.l_jetEta[_iEvent][iJet], self.l_jetPhi[_iEvent][iJet], self.l_jetMass[_iEvent][iJet])
                if iJet not in self.jetVectorDict.keys():
                    self.jetVectorDict[iJet] = _tlv_jet
        
                # continue if jet not within deltaR cone of quark
                if _tlv_quark.delta_r( _tlv_jet) > self.dR_cut_quarkToJet:
                    continue

                if iQuark not in self.matchedQuarksToJets.keys():
                    self.matchedQuarksToJets.update({iQuark:[iJet]})
                else:
                    self.matchedQuarksToJets[iQuark].append(iJet)

        return 


    ##############################################################
    ##                FUNCTIONS FOR MATCHING                    ##
    ##############################################################

    def getHarmonicMeanDeltaR( self, _jetPairTuple, _jetVectorDict):

        # get deltaR between each pair
        _deltaR_pair1 = _jetVectorDict[_jetPairTuple[0]].delta_r(_jetVectorDict[_jetPairTuple[1]])
        _deltaR_pair2 = _jetVectorDict[_jetPairTuple[2]].delta_r(_jetVectorDict[_jetPairTuple[3]])
        if _deltaR_pair1 == 0 or _deltaR_pair2==0:
            print('pair1',_deltaR_pair1, _jetPairTuple[0], _jetPairTuple[1], _jetVectorDict[_jetPairTuple[0]].pt, _jetVectorDict[_jetPairTuple[1]].pt)
            print('pair2',_deltaR_pair2, _jetPairTuple[2], _jetPairTuple[3], _jetVectorDict[_jetPairTuple[2]].pt, _jetVectorDict[_jetPairTuple[3]].pt)
            
        # calculate the harmonic mean
        _meanDeltaR = np.reciprocal( ( np.reciprocal(_deltaR_pair1) + np.reciprocal(_deltaR_pair2) ) / 2 )
        #print(_jetPairTuple, _meanDeltaR, _deltaR_pair1, _deltaR_pair2)

        return _meanDeltaR

    def getEqualDeltaR( self, _jetPairTuple, _jetVectorDict):

        # get deltaR between each pair
        _deltaR_pair1 = _jetVectorDict[_jetPairTuple[0]].delta_r(_jetVectorDict[_jetPairTuple[1]])
        _deltaR_pair2 = _jetVectorDict[_jetPairTuple[2]].delta_r(_jetVectorDict[_jetPairTuple[3]])
        
        # calculate the harmonic mean
        _diffDeltaR = abs( _deltaR_pair1 - _deltaR_pair2 )
        #print(_jetPairTuple, _meanDeltaR, _deltaR_pair1, _deltaR_pair2)
        
        return _diffDeltaR


    def getHiggsMassDifference( self, _jetPairTuple, _jetVectorDict):

        # get deltaR between each pair
        _mass_pair1 = ( _jetVectorDict[_jetPairTuple[0]] + _jetVectorDict[_jetPairTuple[1]] ).mass 
        _mass_pair2 = ( _jetVectorDict[_jetPairTuple[2]] + _jetVectorDict[_jetPairTuple[3]] ).mass 
        
        # calculate the quadrature sum of higgs mass diff and divide by reco higgs width
        _quadratureMassDifference = np.sqrt( ( (_mass_pair1 - self.mass_higgs) / self.width_higgs )**2 + ( (_mass_pair2 - self.mass_higgs) / self.width_higgs )**2 )
        #print(_jetPairTuple, _quadratureMassDifference, _massDiff_pair1, _massDiff_pair2)
        
        return _quadratureMassDifference


    def getDijetMassDifference( self, _jetPairTuple, _jetVectorDict):

        # get masses for each pair
        _mass_pair1 = ( _jetVectorDict[_jetPairTuple[0]] + _jetVectorDict[_jetPairTuple[1]] ).mass 
        _mass_pair2 = ( _jetVectorDict[_jetPairTuple[2]] + _jetVectorDict[_jetPairTuple[3]] ).mass 
        
        # calculate the direct difference of reco dijet masses
        _dijetMassDifference = abs(_mass_pair1 - _mass_pair2)
        #print(_jetPairTuple, _quadratureMassDifference, _massDiff_pair1, _massDiff_pair2)
        
        return _dijetMassDifference


    def getBothDijetMasses( self, _jetPairTuple, _jetVectorDict):
        
        # get masses of each pair
        _mass_pair1 = ( _jetVectorDict[_jetPairTuple[0]] + _jetVectorDict[_jetPairTuple[1]] ).mass 
        _mass_pair2 = ( _jetVectorDict[_jetPairTuple[2]] + _jetVectorDict[_jetPairTuple[3]] ).mass 
        
        # make a list of the two masses
        _bothDijetMasses = [_mass_pair1, _mass_pair2 ]
        
        return _bothDijetMasses
        
        
    def returnMetric( self, _pairingAlgorithm, _sortedTuple, _jetVectorDict ):
        # calculate metric depending on chosen algorithm
        _metric = []
        
        if _pairingAlgorithm == "minHarmonicMeanDeltaR":
            _metric = self.getHarmonicMeanDeltaR(_sortedTuple, _jetVectorDict)
        elif _pairingAlgorithm == "closestDijetMassesToHiggs":
            _metric = self.getHiggsMassDifference(_sortedTuple, _jetVectorDict)
        elif _pairingAlgorithm == "equalDijetMass":
            _metric = self.getDijetMassDifference(_sortedTuple, _jetVectorDict)
        elif _pairingAlgorithm == "equalDeltaR":
            _metric = self.getEqualDeltaR(_sortedTuple, _jetVectorDict)
        elif _pairingAlgorithm == "dijetMasses":
            _metric = self.getBothDijetMasses(_sortedTuple, _jetVectorDict)
            
        _metric = _metric if type(_metric)==list else [_metric]
        return _metric


    def selectPairsViaMatchingAlgorithm( self, _pairingAlgorithm):
        
        # *** 0. Make dict of jets to analyze
        _jetsToAnalyze = {}
        _jetsToAnalyze = self.jetVectorDict
        """if self.considerFirstNjetsInPT==-1:
            # use all jets
            _jetsToAnalyze = self.jetVectorDict
        else:         
            # choose subset of jets if required by user
            _firstNjets = list(sorted(self.jetVectorDict.keys()))[:self.considerFirstNjetsInPT]
            for iJet in _firstNjets:
        _jetsToAnalyze[iJet] = self.jetVectorDict[iJet]
        """
            
        # *** 1. Make list of pairs from [n choose 2] where n is number of jets
        _jetPairs = list(itertools.combinations(_jetsToAnalyze.keys(),2))
        _doubleJetPairs = {}

        # *** 2. Loop over jet pairs
        for pair in _jetPairs:
            # ** A. Make list of leftover pairs that do not contain either jet in starting pair
            _notPair = [x for x in list(_jetPairs) if pair[0] not in x and pair[1] not in x]
            for pairOption in _notPair:
                _sortedPairing = sorted([sorted(x) for x in [pair, pairOption]])
                _sortedTuple = tuple(_sortedPairing[0]+_sortedPairing[1])

                # ** B. Add double pairing to dictionary if not already present. sorting removes positional ambiguity
                if _sortedTuple not in _doubleJetPairs.keys():
                    _metric = self.returnMetric(_pairingAlgorithm, _sortedTuple, _jetsToAnalyze)

                    _doubleJetPairs[_sortedTuple] = _metric
                    self.plottingData[_pairingAlgorithm]['All'].extend( _metric )
                    if self.thisEventIsMatchable:
                        self.plottingData[_pairingAlgorithm]['Matchable'].extend( _metric )

        # ** C. Sort output dict and find minimal value
        _bestPairing = sorted(_doubleJetPairs.items(), key=lambda _pairingAndMetric: _pairingAndMetric[1][0])[0]
        self.plottingData[_pairingAlgorithm]['Best'].extend( _bestPairing[1] )   
        if self.thisEventIsMatchable:
            # ** D. Fill algorithm-selected lists for plotting
            self.plottingData[_pairingAlgorithm]['Best+Matchable'].extend( _bestPairing[1] )


        return (_bestPairing[0][0], _bestPairing[0][1]), (_bestPairing[0][2] , _bestPairing[0][3]), _bestPairing[1][0]


    def fillVariablePlotsForCorrectPairing( self, iEvt, _matchedJetVector, _pairingAlgorithm):
        _correctTuple = (0, 1, 2, 3)
        _metric = self.returnMetric(_pairingAlgorithm, _correctTuple, _matchedJetVector)
        self.plottingData[_pairingAlgorithm]['Correct'].extend( _metric )
        if _metric[0]==0:
            print (iEvt)

        return

    
    ##############################################################
    ##                FUNCTIONS FOR EFFICIENCY                  ##
    ##############################################################

    def returnJetTagLabels( self ):

        # every event is inclusive
        _categoryLabels = ['Incl']    
    
        # split into tag-inclusive bins, 6j means >= 6 jets
        if self.nJets == 4:
            _categoryLabels.append('4jIncl')
        elif self.nJets == 5:
            _categoryLabels.append('5jIncl')
        elif self.nJets == 6:
            _categoryLabels.append('6jIncl')
        elif self.nJets >= 7:
            _categoryLabels.append('7jIncl')
            
        # split into tag bins, 4b means >= 4 tags
        _jetLabel = str(self.nJets) if self.nJets <= 7 else str(7)
        _tagLabel = str(self.nBTags) if self.nBTags <= 7 else str(7)
        _categoryLabels.append( _jetLabel+'j'+_tagLabel+'b' )
        
        return _categoryLabels


    def countEvents( self, _cutflowBin ):

        _categoryLabels = self.returnJetTagLabels()
        for iAlgorithm in self.eventCounterDict:
            for iLabel in _categoryLabels:
                self.eventCounterDict[iAlgorithm][iLabel][_cutflowBin] += 1
        
    
    def evaluatePairingEfficiency( self, _jetPair1, _jetPair2, _algorithm):
                
        # Organize quark-to-jet pairs from truth into directly comparable tuples
        _indexList = list( self.matchedQuarksToJets.values() ) 
        _orderedIndexTuple = sorted( ( tuple(sorted( (_indexList[0][0], _indexList[1][0]) )) , tuple(sorted( (_indexList[2][0], _indexList[3][0]) )) ) )
        _indexPair1 = _orderedIndexTuple[0]
        _indexPair2 = _orderedIndexTuple[1]
        
        # Do some global counting
        _categoryLabels = self.returnJetTagLabels()
        for iLabel in _categoryLabels:
            if _jetPair1 == _indexPair1 and _jetPair2 == _indexPair2:
                self.eventCounterDict[_algorithm][iLabel]['Fully Matched'] += 1
         
            if _jetPair1 == _indexPair1 or _jetPair2 == _indexPair2:
                self.eventCounterDict[_algorithm][iLabel]['>= 1 Pair Matched'] += 1
        
        return 


        
    def printEventCounterInfo( self, _algorithm, _catTag ):
        print('====================================================')
        print("!!!! Event Counter Info For " + _algorithm + ", " + _catTag)
        print("Number of Events:", self.eventCounterDict[_algorithm][_catTag]['All'])
        print("Number of Events with 4 truth-matchable jets:", self.eventCounterDict[_algorithm][_catTag]['Matchable'])
        print("Number of Events Fully Matched:", self.eventCounterDict[_algorithm][_catTag]['Fully Matched'])
        print("Number of Events with >= 1 Pair Matched:", self.eventCounterDict[_algorithm][_catTag]['>= 1 Pair Matched'])
        if self.eventCounterDict[_algorithm][_catTag]['Matchable'] > 0:
            print('Efficiency For Fully Matched: ',round( 100*float(self.eventCounterDict[_algorithm][_catTag]['Fully Matched']/self.eventCounterDict[_algorithm][_catTag]['Matchable']) , 2),'%')
            print('Efficiency For >= 1 Pair Matched: ',round( 100*float(self.eventCounterDict[_algorithm][_catTag]['>= 1 Pair Matched']/self.eventCounterDict[_algorithm][_catTag]['Matchable']) , 2),'%')
 
        return


    def listOfEfficienciesForAlgorithm( self ):
        _fullyMatchedDict = {}
        _onePairMatchedDict = {}
        _labels = []

        for _iAlgorithm in self.eventCounterDict:
            _fullyMatchedDict[_iAlgorithm] = {}
            _onePairMatchedDict[_iAlgorithm] = {}
            for _iCategory in self.eventCounterDict[_iAlgorithm]:

                # Calculate efficiencies
                if self.eventCounterDict[_iAlgorithm][_iCategory]['Matchable'] > 0:
                    _fullyMatchedEff  = round( 100*float(self.eventCounterDict[_iAlgorithm][_iCategory]['Fully Matched']/self.eventCounterDict[_iAlgorithm][_iCategory]['Matchable']), 2)
                    _onePairMatchedEff = round( 100*float(self.eventCounterDict[_iAlgorithm][_iCategory]['>= 1 Pair Matched']/self.eventCounterDict[_iAlgorithm][_iCategory]['Matchable']), 2)
                    _fullyMatchedDict[_iAlgorithm][_iCategory] = _fullyMatchedEff
                    _onePairMatchedDict[_iAlgorithm][_iCategory] = _onePairMatchedEff

                    # ** Make list of jet/tag categories
                    if( len(_fullyMatchedDict.keys())<2 ):
                        _labels.append(_iCategory)

                    #print('Efficiency For Fully Matched: ', _fullyMatchedEff,'%')
                    #print('Efficiency For >= 1 Pair Matched: ', _onePairMatchedEff,'%')
                
        return _fullyMatchedDict, _onePairMatchedDict, _labels


    def calculateAndDrawEfficiency( self, _matchedDict, _labels, _onePairOrFullyMatched ):

        _xvals = np.arange(len(_labels))
        plt.xticks(range(len(_labels)), _labels, rotation=45)
        
        for algo in _matchedDict:
            if algo != 'dijetMasses':
                _eff_algo = [_matchedDict[algo][tagEff] for tagEff in _matchedDict[algo]]
                _y_algo = np.array(_eff_algo)
                plt.plot(_xvals, _y_algo, label=algo)

        plt.ylim(-18, 105)
        plt.legend(loc='lower left')
        
        # store figure copy for later saving
        fig = plt.gcf()
        
        # draw interactively
        plt.show()
        
        fig.savefig( self.datasetName + '/algoEfficiencyAsFtnOfJetTag_' + _onePairOrFullyMatched +'.png' )


    ##############################################################
    ##           FUNCTIONS FOR CALCULATING BDT VARS             ##
    ##############################################################

    def createOutputVariableList( self ):
        _variableNameList = ['hh_mass', 'h1_mass', 'h2_mass', 'hh_pt', 'h1_pt', 'h2_pt', 'deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'deltaPhi(h1, h2)', 'deltaPhi(h1 jets)', 'deltaPhi(h2 jets)', 'met', 'met_phi', 'scalarHT', 'nJets', 'nBTags', 'isMatchable']
        _jetVariables = ['pt', 'eta', 'phi', 'mass', 'px', 'py', 'pz', 'energy', 'btag']
        _genVariables = ['PID', 'status', 'pt', 'eta', 'phi', 'mass']
    
        for _variable in _jetVariables:
            _variableNameList.extend( ['recoJet'+str(_iJet)+'_'+str(_variable) for _iJet in range(1,5)]) # jets used in selected dihiggs reconstruction (should always be 4)

        for _variable in _jetVariables:
            _variableNameList.extend( ['jet'+str(_iJet)+'_'+str(_variable) for _iJet in range(1,self.nJetsToStore+1)]) # for all jets by pT ... b-tagged first and then pt ordered

        for _variable in _genVariables:
            _variableNameList.extend( ['gen'+str(_iJet)+'_'+str(_variable) for _iJet in range(1, 5)]) # for some relevant subset of generated jets ... should be max 4(+1) for the four b quarks from h->bb x 2

        return _variableNameList


    def calculateVariablesForBDT( self, _iEvent, _jetPair1, _jetPair2 ):
        _variableList = []
    
        _tlv_h1_j0 = self.jetVectorDict[ _jetPair1[0] ]
        _tlv_h1_j1 = self.jetVectorDict[ _jetPair1[1] ]
        _tlv_h2_j2 = self.jetVectorDict[ _jetPair2[0] ]
        _tlv_h2_j3 = self.jetVectorDict[ _jetPair2[1] ]
        _tlv_h1 = _tlv_h1_j0 + _tlv_h1_j1
        _tlv_h2 = _tlv_h2_j2 + _tlv_h2_j3

        """print('====================================================')
        print ("hh mass: ", (_tlv_h1 + _tlv_h2).mass)
        print ("h1 mass: ", _tlv_h1.energy, _tlv_h1.p, _tlv_h1_j0.energy, _tlv_h1_j0.p, _tlv_h1_j1.energy, _tlv_h1_j1.p)
        print ("h2 mass: ", _tlv_h2.mass)
        print ("hh pt: ", (_tlv_h1 + _tlv_h2).pt)
        print ("h1 pt: ", _tlv_h1.pt)
        print ("h2 pt: ", _tlv_h2.pt)
        print ("dR(h1, h2): ",  _tlv_h1.delta_r( _tlv_h2 ))
        print ("for h1, dR(j0, j1): ",  _tlv_h1_j0.delta_r( _tlv_h1_j1 ))
        print ("for h2, dR(j2, j3): ",  _tlv_h2_j2.delta_r( _tlv_h2_j3 ))
        print ("for h1, dPhi(j0, j1): ",  _tlv_h1_j0.delta_phi( _tlv_h1_j1 ))
        print ("for h2, dPhi(j2, j3): ",  _tlv_h2_j2.delta_phi( _tlv_h2_j3 ))
        #print ("MET, met_phi: ", _met[0], _met_phi[0])
        #print ("Scalar HT: ", _scalarHT[0])
        #print ("nJets, nBTags: ", self.nJets, self.nBTags)
        """
        _nDigits = 3
        
        _variableList = [ (_tlv_h1 + _tlv_h2).mass, _tlv_h1.mass, _tlv_h2.mass,
                          (_tlv_h1 + _tlv_h2).pt, _tlv_h1.pt, _tlv_h2.pt,
                          _tlv_h1.delta_r(_tlv_h2), 
                          _tlv_h1_j0.delta_r(_tlv_h1_j1), _tlv_h2_j2.delta_r(_tlv_h2_j3), 
                          _tlv_h1.delta_phi(_tlv_h2), 
                          _tlv_h1_j0.delta_phi(_tlv_h1_j1), _tlv_h2_j2.delta_phi(_tlv_h2_j3), 
                          self.l_missingET_met[_iEvent][0], self.l_missingET_phi[_iEvent][0], self.l_scalarHT[_iEvent][0], 
                          self.nJets, self.nBTags,
                          self.thisEventIsMatchable ]
        if self.saveLowLevelVariablesForTraining==True:
            # jets used in dihiggs reconstruction
            _variableList.extend( [_tlv_h1_j0.pt, _tlv_h1_j1.pt, _tlv_h2_j2.pt, _tlv_h2_j3.pt, 
                                   _tlv_h1_j0.eta, _tlv_h1_j1.eta, _tlv_h2_j2.eta, _tlv_h2_j3.eta,
                                   _tlv_h1_j0.phi, _tlv_h1_j1.phi, _tlv_h2_j2.phi, _tlv_h2_j3.phi,
                                   _tlv_h1_j0.mass, _tlv_h1_j1.mass, _tlv_h2_j2.mass, _tlv_h2_j3.mass,
                                   _tlv_h1_j0.x, _tlv_h1_j1.x, _tlv_h2_j2.x, _tlv_h2_j3.x, 
                                   _tlv_h1_j0.y, _tlv_h1_j1.y, _tlv_h2_j2.y, _tlv_h2_j3.y, 
                                   _tlv_h1_j0.z, _tlv_h1_j1.z, _tlv_h2_j2.z, _tlv_h2_j3.z, 
                                   _tlv_h1_j0.energy, _tlv_h1_j1.energy, _tlv_h2_j2.energy, _tlv_h2_j3.energy,
                                   self.l_jetBTag[_iEvent][_jetPair1[0]], self.l_jetBTag[_iEvent][_jetPair1[1]], self.l_jetBTag[_iEvent][_jetPair2[0]], self.l_jetBTag[_iEvent][_jetPair2[1]]
                               ] )

            # save info on all jets up to self.nJetsToStore
            allJetVectors = [ TLorentzVector.PtEtaPhiMassLorentzVector( self.l_jetPt[_iEvent][iJet], self.l_jetEta[_iEvent][iJet], self.l_jetPhi[_iEvent][iJet], self.l_jetMass[_iEvent][iJet]) for iJet in self.allJetIndices ]
            allJetBTags   = [ self.l_jetBTag[_iEvent][iJet] for iJet in self.allJetIndices ]
            allJetVectors = allJetVectors[:self.nJetsToStore] if len(allJetVectors) > self.nJetsToStore else allJetVectors
            allJetBTags   = allJetBTags[:self.nJetsToStore] if len(allJetBTags) > self.nJetsToStore else allJetBTags
            while len(allJetVectors) < self.nJetsToStore:
                allJetVectors.append( TLorentzVector.PtEtaPhiMassLorentzVector(0, 0, 0, 0) ) # zero-padding
                allJetBTags.append( -1 ) # zero-padding

            allJetPt     = [ i_tlv.pt for i_tlv in allJetVectors]
            allJetEta    = [ i_tlv.eta for i_tlv in allJetVectors]
            allJetPhi    = [ i_tlv.phi for i_tlv in allJetVectors]
            allJetMass   = [ i_tlv.mass for i_tlv in allJetVectors]
            allJetPx     = [ i_tlv.x for i_tlv in allJetVectors]
            allJetPy     = [ i_tlv.y for i_tlv in allJetVectors]
            allJetPz     = [ i_tlv.z for i_tlv in allJetVectors]
            allJetEnergy = [ i_tlv.energy for i_tlv in allJetVectors]

            _variableList.extend( allJetPt )
            _variableList.extend( allJetEta )
            _variableList.extend( allJetPhi )
            _variableList.extend( allJetMass )
            _variableList.extend( allJetPx )
            _variableList.extend( allJetPy )
            _variableList.extend( allJetPz )
            _variableList.extend( allJetEnergy )
            _variableList.extend( allJetBTags )

            # save info on quarks if relevant
            allGenVectors = [ TLorentzVector.PtEtaPhiMassLorentzVector( self.l_genPt[_iEvent][iQuark], self.l_genEta[_iEvent][iQuark], self.l_genPhi[_iEvent][iQuark], self.l_genMass[_iEvent][iQuark]) for iQuark in self.quarkIndices ]
            allGenPID     = [ self.l_genPID[_iEvent][iQuark] for iQuark in self.quarkIndices ]
            allGenStatus  = [ self.l_genStatus[_iEvent][iQuark] for iQuark in self.quarkIndices ]
            allGenVectors = allJetVectors[:4] if len(allGenVectors) > 4 else allGenVectors

            while len(allGenVectors) < 4:
                allGenVectors.append( TLorentzVector.PtEtaPhiMassLorentzVector(0, 0, 0, 0) ) # zero-padding
                allGenPID.append( -1 ) # zero-padding, not sure if this is the right move for status or PID
                allGenStatus.append( -1 ) # zero-padding, not sure if this is the right move for status or PID

            allGenPt     = [ i_tlv.pt for i_tlv in allGenVectors]
            allGenEta    = [ i_tlv.eta for i_tlv in allGenVectors]
            allGenPhi    = [ i_tlv.phi for i_tlv in allGenVectors]
            allGenMass   = [ i_tlv.mass for i_tlv in allGenVectors]

            _variableList.extend( allGenPID )
            _variableList.extend( allGenStatus )
            _variableList.extend( allGenPt )
            _variableList.extend( allGenEta )
            _variableList.extend( allGenPhi )
            _variableList.extend( allGenMass )

            
        return _variableList



    ##############################################################
    ##           FUNCTIONS FOR RUNNING RECONSTRUCTION           ##
    ##############################################################


    def truthToRecoMatching( self, _iEvent, ):

        self.thisEventIsMatchable = False
        self.thisEventWasCorrectlyMatched = False
        self.matchedQuarksToJets = {}
        self.jetVectorDict = {}
        self.quarkVectorDict = {}

        # Return if QCD --> no truth to assign
        if self.isDihiggsMC == False:

            for iJet in self.jetIndices:           
                # make vector for jets
                _tlv_jet = TLorentzVector.PtEtaPhiMassLorentzVector( self.l_jetPt[_iEvent][iJet], self.l_jetEta[_iEvent][iJet], self.l_jetPhi[_iEvent][iJet], self.l_jetMass[_iEvent][iJet])
                if iJet not in self.jetVectorDict.keys():
                    self.jetVectorDict[iJet] = _tlv_jet
            return

        self.jetCounter[str(len(self.jetVectorDict))] += 1
        self.getDictOfQuarksMatchedToJets( _iEvent )
        self.matchedCounter[str(len(self.jetVectorDict))] += 1

        # Check if a) all matches have one and only match between quark and jet, b) four jets are matched, c) 4 unique reconstructed jets are selected
        _jetIndexList = [recoIndex[0] for recoIndex in self.matchedQuarksToJets.values()]
        if all(len(matchedJets) == 1 for matchedJets in self.matchedQuarksToJets.values()) and len(self.matchedQuarksToJets)==4  and (len(set(_jetIndexList)) == len(_jetIndexList)):     
            self.thisEventIsMatchable = True
            self.countEvents( 'Matchable' )


        return 


    def getRecoInformation( self, _iEvent ):
        self.returnNumberAndListOfJetIndicesPassingCuts( _iEvent )
        if(self.saveJetConstituents==True):
            self.outputJetCons( _iEvent )
        self.nJetsPerEvent.append( self.nJets )
        self.nBTagsPerEvent.append( self.nBTags  )
        self.countEvents( 'All' )
        
        return 

    def getTruthInformation( self, _iEvent ):

        self.quarkIndices = []       
        # Return if QCD --> no truth to assign
        if self.isDihiggsMC == False:
            return

        self.returnListOfTruthBQuarkIndicesByStatus( _iEvent )   
        #self.returnListOfTruthBQuarkIndicesByDaughters( _iEvent )   

        if len( self.quarkIndices ) != 4:
            print ("!!! WARNING: Event = {0} did not find 4 truth b-quarks. Only found {1} !!!".format(iEvent, len(self.quarkIndices)))
            
        return 

    def getJetConsCand( self ):
        Cons_list = []
        #obj_name = ["EFlowTrack", "EFlowNeutralHadron", "EFlowPhoton"]
        #obj_keys = ["fUniqueID", "PT", "Eta", "Phi","P"]
        #Cons_list[i] is feature for object obj_name[i]
        #Cons_list[iObj][iKey][iEvt][iCons]
        for obj in self.jetConsCand_Names:
            #print(obj)
            obj_values = []
            for key in self.jetConsCand_Keys:
                if(not obj=="EFlowTrack"):
                    if( key=="PT"):
                        key = "ET"
                    elif(key=="P"):
                        key = "E"
                obj_values.append(self.delphesFile[obj]["{0}.{1}".format(obj,key)].array())
            Cons_list.append(obj_values)
        return Cons_list

    def getListOfJetCons( self, _iEvent ):
        ConsList = []
        #print("For Event: {}".format(_iEvent))
        for iJet in self.allJetIndices:
            #print("For Jet: {0} PT: {1} Eta: {2} Phi: {3} ".format(iJet,self.l_jetPt[_iEvent][iJet],self.l_jetEta[_iEvent][iJet],self.l_jetPhi[_iEvent][iJet]))
            for iObj_t in range(0,len(self.jetConsCand)):     #loop over all types of object s.t. tracks towers etc.
                mask = np.isin(self.jetConsCand[iObj_t][0][_iEvent],self.l_jetCons[_iEvent][iJet])    #use a mask to pick all the candidats that are present in jet
                UID_array = self.jetConsCand[iObj_t][0][_iEvent][mask]
                PT_array = self.jetConsCand[iObj_t][1][_iEvent][mask]
                Eta_array = self.jetConsCand[iObj_t][2][_iEvent][mask]
                Phi_array = self.jetConsCand[iObj_t][3][_iEvent][mask]
                E_array = self.jetConsCand[iObj_t][4][_iEvent][mask]
                for iObj in range(0,len(UID_array)):
                    ConsList.append(
                        JetCons(UID_array[iObj], #UID
                                PT_array[iObj],  #PT
                                Eta_array[iObj], #Eta
                                Phi_array[iObj], #Phi
                                E_array[iObj],   #E
                        )
                    )
                    ConsList[-1].property['type'] = iObj_t # 0 == EFlowTrack, 1 == EFlowNeutralHadron, 2 == EFlowPhoton

                #print("UID: {0}  PT: {1}  Eta: {2}  Phi: {3}".format(self.jetConsCand[iObj_t][0][_iEvent][iObj],self.jetConsCand[iObj_t][1][_iEvent][iObj],self.jetConsCand[iObj_t][2][_iEvent][iObj],self.jetConsCand[iObj_t][3][_iEvent][iObj]))
        #self.JetConsList.append(ConsList)
        return ConsList

    def outputJetCons( self, _iEvent ):
        ConsList = self.getListOfJetCons( _iEvent )
        cons = []
        for iCons in ConsList:
            cons_properties = []
            for key in self.jetConsOutputKeys:
                cons_properties.append(iCons.get(key))
            cons.append(np.array(cons_properties))
        self.outputJetConsInfo.append(np.array(cons))
        return


    

    def returnJetVectorInputsAsBranch( self, _df, _jetType='jet', _jetVar='px', _nJets=4):
    
        #flattened
        _allVectorsFlattened = None
        _var = [_jetType+'{}_'+_jetVar]
    
        for i in range(1, _nJets + 1):
            _varN = [x.format(i) for x in _var]
            _jetNData = _df[ _varN ].astype(np.float32)
            _vectN = [list(x) for x in _jetNData.values]
        
            if _allVectorsFlattened == None:
                _allVectorsFlattened = _vectN
            else:
                _allVectorsFlattened = [ x + y for x,y in zip(_allVectorsFlattened, _vectN) ]

        
        return _allVectorsFlattened


    
    def initFileAndBranches( self ):

        if '.root' in self.inputFileName:
            print("Setting Delphes file; ", self.inputFileName)
            self.delphesFile= uproot.open(self.inputFileName)['Delphes']
            self.nEntries = self.delphesFile._fEntries
            
            # branches
            self.l_genPID         = uproot.tree.TBranchMethods.array(self.delphesFile['Particle']['Particle.PID']).tolist()
            self.l_genStatus      = uproot.tree.TBranchMethods.array(self.delphesFile['Particle']['Particle.Status']).tolist()
            self.l_genPt          = uproot.tree.TBranchMethods.array(self.delphesFile['Particle']['Particle.PT']).tolist()
            self.l_genEta         = uproot.tree.TBranchMethods.array(self.delphesFile['Particle']['Particle.Eta']).tolist()
            self.l_genPhi         = uproot.tree.TBranchMethods.array(self.delphesFile['Particle']['Particle.Phi']).tolist()
            self.l_genMass        = uproot.tree.TBranchMethods.array(self.delphesFile['Particle']['Particle.Mass']).tolist()
            self.l_jetPt          = uproot.tree.TBranchMethods.array(self.delphesFile['Jet']['Jet.PT']).tolist()
            self.l_jetEta         = uproot.tree.TBranchMethods.array(self.delphesFile['Jet']['Jet.Eta']).tolist()
            self.l_jetPhi         = uproot.tree.TBranchMethods.array(self.delphesFile['Jet']['Jet.Phi']).tolist()
            self.l_jetMass        = uproot.tree.TBranchMethods.array(self.delphesFile['Jet']['Jet.Mass']).tolist()
            self.l_jetBTag        = uproot.tree.TBranchMethods.array(self.delphesFile['Jet']['Jet.BTag']).tolist()
            self.l_jetCons        = uproot.tree.TBranchMethods.array(self.delphesFile['Jet']['Jet.Constituents']).tolist()
            self.l_missingET_met  = uproot.tree.TBranchMethods.array(self.delphesFile['MissingET']['MissingET.MET']).tolist()
            self.l_missingET_phi  = uproot.tree.TBranchMethods.array(self.delphesFile['MissingET']['MissingET.Phi']).tolist()
            self.l_scalarHT       = uproot.tree.TBranchMethods.array(self.delphesFile['ScalarHT']['ScalarHT.HT']).tolist()
            self.jetConsCand      = self.getJetConsCand()
            print("Finished loading branches...")

        elif '.csv' in self.inputFileName:
            print("Setting pre-proceseed .csv file; ", self.inputFileName)
            self.inputDF = pd.read_csv(self.inputFileName)
            hh_raw = pd.read_csv('../../dihiggsMLProject_copy/data/pp2hh4b_CMSPhaseII_0PU_top4Tags_store8jets/dihiggs_outputDataForLearning_pp2hh4b_CMSPhaseII_0PU_top4Tags_store8jets.csv')
            self.nEntries = len(self.inputDF[ self.inputDF.columns[0] ])
            _jetType='jet' # this just takes all the jets, this is probably fine to leave alone but maybe needs tweaking if results not identical between delphes output and csv re-process
            _genType='gen' 

            # "branches"
            self.l_genPID         = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _genType, 'PID', 4)
            self.l_genStatus      = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _genType, 'status', 4)
            self.l_genPt          = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _genType, 'pt', 4)
            self.l_genEta         = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _genType, 'eta', 4)
            self.l_genPhi         = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _genType, 'phi', 4)
            self.l_genMass        = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _genType, 'mass', 4)
            self.l_jetPt          = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _jetType, 'pt', self.nJetsToStore)
            self.l_jetEta         = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _jetType, 'eta', self.nJetsToStore)
            self.l_jetPhi         = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _jetType, 'phi', self.nJetsToStore)
            self.l_jetMass        = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _jetType, 'mass', self.nJetsToStore)
            self.l_jetBTag        = self.returnJetVectorInputsAsBranch( self.inputDF.copy(), _jetType, 'btag', self.nJetsToStore)
            self.l_missingET_met  = [ [x] for x in self.inputDF.met.tolist()]
            self.l_missingET_phi  = [ [x] for x in self.inputDF.met_phi.tolist()]
            self.l_scalarHT       = [ [x] for x in self.inputDF.scalarHT.tolist()]
            
            print("Finished loading branches...")



            
    def evaluatePairingAlgorithms( self, _iEvent ): 

        for iAlgorithm in self.pairingAlgorithms:
            # ** A. Fill algorithm metric for correct pairing (regardless if chosen by metric)
            if self.thisEventIsMatchable == True:
                self.fillVariablePlotsForCorrectPairing( _iEvent, [self.jetVectorDict[matchedJet[0]] for matchedJet in self.matchedQuarksToJets.values()], iAlgorithm)
            
            # ** B. Pick two jet pairs based on algorithm
            _jetPair1, _jetPair2, _pairingMetric = self.selectPairsViaMatchingAlgorithm( iAlgorithm )
    
            # ** C. Evaluate efficiency of pairing algorithm
            if self.thisEventIsMatchable:
                self.evaluatePairingEfficiency( _jetPair1, _jetPair2, iAlgorithm)
    
            # ** D. Calculate and save variables for BDT training for single algorithm set by saveAlgorithm
            if iAlgorithm == self.saveAlgorithm: 
                _variablesForBDT = self.calculateVariablesForBDT(_iEvent, _jetPair1, _jetPair2)
                self.outputDataForLearning.append(_variablesForBDT)

        return


    def writeDataForTraining(self):

        _csvName = self.datasetName + '/' + ('dihiggs_' if self.isDihiggsMC else 'qcd_') + 'outputDataForLearning_{0}_{1}.csv'.format(self.datasetName,self.saveIdx)
        _csvFile = open(_csvName, mode='w')
        _writer = csv.DictWriter(_csvFile, fieldnames=self.outputVariableNames)
        _writer.writeheader()

        
        for eventData in self.outputDataForLearning:
            _csvLineDict = {}
            # format csv line using names of variables
            for iVariable in range(0, len(self.outputVariableNames)):
                _csvLineDict[ self.outputVariableNames[iVariable] ] = eventData[iVariable]
            # write line to .csv
            _writer.writerow( _csvLineDict )
        _csvFile.close()
       
        return

    def writeJetConsData(self):
        _savename = self.datasetName + '/' + ('dihiggs_' if self.isDihiggsMC else 'qcd_') + 'outputJetData_{0}_{1}.pkl'.format(self.datasetName,self.saveIdx)
        file_to_save = open(_savename, "wb")
        pickle.dump(self.outputJetConsInfo, file_to_save, -1)
        file_to_save.close()

    
