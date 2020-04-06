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

    def __init__ (self, _datasetName, _inputFileList, _isSignal, _isTestRun=False):
        self.datasetName = _datasetName

        if os.path.isfile(_inputFileList):
            with open(_inputFileList, 'r') as f:
                self.inputFileList = f.readlines()
        else:
            print("!!! Input file {} DNE !!!".format(_inputFileList) )

        self.isSignal = _isSignal
        self.isTestRun = _isTestRun
        if os.path.isdir( self.datasetName )==False:
            os.mkdir( self.datasetName )
        self.pixelWidth = 15
            
        ## Objects per file
        # input file
        self.allEvents = []

        # raw for processing (stored by event)
        self.tracks = [ ]
        self.nHadrons = [ ]
        self.photons = [ ]
        self.nJets = [ ]
        self.nBTags = [ ]

        # final stored by event and with boost/rotation
        self.final_tracks = [ ]
        self.final_nHadrons = [ ]
        self.final_photons = [ ]

        ## Objects per dataset
        # images for saving
        self.final_images = [ [], [], [], [], [] ] # tracks, nHadrons, photons, composite (all three), composite (only events with >=4j)
        self.final_eventQuantities = [ [], [] ] # nJets, nBTags


    ##############################################################
    ##                           MAIN                           ##
    ##############################################################

    def processFiles(self):
        """ central function call body for producing images"""

        for infile in self.inputFileList:

            print(infile)
            # *** 0. Load events and constituents
            self.loadEventsAndMakeContainers( infile )

            # *** 1. Make some overall plots (comment out as necessary)
            #self.quickPlots()
            self.rawDetectorImages()

            # *** 2. Rotate/Boost constituents into desired frame
            self.rotateAndBoostConstituents()

            # *** 3. Make images
            self.makeEventImages()


        # *** 4. Save images from all processed files
        self.saveFilesToH5()

        
        return
    
    ##############################################################
    ##                        FUNCTIONS                         ##
    ##############################################################

    def rawDetectorImages(self):
        """ raw detector images without boosting or rotating. should be nonsense, but useful for comparison"""
        _allTracks_phi   = [ track_phi for event in self.tracks for track_phi in event[0] ]
        _allTracks_rap   = [ track_rap for event in self.tracks for track_rap in event[1] ]
        _allTracks_pt    = [ track_pt for event in self.tracks for track_pt in event[2] ]

        _allNHadrons_phi = [ nHadron_phi for event in self.nHadrons for nHadron_phi in event[0] ]
        _allNHadrons_rap = [ nHadron_rap for event in self.nHadrons for nHadron_rap in event[1] ]
        _allNHadrons_pt  = [ nHadron_pt for event in self.nHadrons for nHadron_pt in event[2] ]
        
        _allPhotons_phi = [ photon_phi for event in self.photons for photon_phi in event[0] ]
        _allPhotons_rap = [ photon_rap for event in self.photons for photon_rap in event[1] ]
        _allPhotons_pt  = [ photon_pt for event in self.photons for photon_pt in event[2] ]

        # Raw-raw, i.e. no pT weighting
        self.makeSimplePlots( [_allTracks_phi, _allTracks_rap, _allTracks_pt], 'Tracks', False)
        self.makeSimplePlots( [_allNHadrons_phi, _allNHadrons_rap, _allNHadrons_pt], 'Neutral Hadrons', False)
        self.makeSimplePlots( [_allPhotons_phi, _allPhotons_rap, _allPhotons_pt], 'Photons', False)

        # Semi-raw, i.e. no boost/rotation but using pT weighting
        self.makeSimplePlots( [_allTracks_phi, _allTracks_rap, _allTracks_pt], 'Tracks', True)
        self.makeSimplePlots( [_allNHadrons_phi, _allNHadrons_rap, _allNHadrons_pt], 'Neutral Hadrons', True)
        self.makeSimplePlots( [_allPhotons_phi, _allPhotons_rap, _allPhotons_pt], 'Photons', True)

        return

    
    def quickPlots(self):
        """quick diagnostic plots that I probably won't use"""

        plt.hist( self.nJets, bins=12, range=[0,12], alpha=0.5, density=1)
        plt.show()
        plt.hist( self.nBTags, bins=7, range=[0,7], alpha=0.5, density=1)
        plt.show()

        return
        
    def loadEventsAndMakeContainers(self, _inputFile):
        """ load events and make track/neutral hadron/photon containers"""

        # *** 0. Declare/Initialize some stuff
        self.allEvents = []
        self.tracks    = []
        self.nHadrons  = []
        self.photons   = []
        self.nJets     = []
        self.nBTags    = []

        # *** 1. Check if file exists and load if yes
        _filename = _inputFile.split('\n')[0]
        print( "Opening file: {} ...".format(_filename))
        if os.path.isfile(_filename):
            # Do something with the file
            with open(_filename, 'rb') as f:
                self.allEvents = pkl.load( f )
            print( "Loading {} events...".format(len(self.allEvents)))
        else:
            print("File not accessible")
            return

        # *** 2. Set jet constituent information
        self.setPhiRapidityPtLists()

        return
    
    def rotateAndBoostConstituents( self ):
        """ Make some event-by-event image plots and also do translation to unified frame """

        self.final_tracks   = [ [], [], [], [] ] # phi, rap, pt, image 
        self.final_nHadrons = [ [], [], [], [] ]
        self.final_photons  = [ [], [], [], [] ]

        _final_tracks   = [ [], [], [] ] # phi, rap, pt
        _final_nHadrons = [ [], [], [] ]
        _final_photons  = [ [], [], [] ]

        for iEvent in range(0, len(self.tracks)):

            # *** Loosely keep track of events
            if 100*((iEvent+1)/len(self.allEvents))%10 == 0:
                print('{}% Processed'.format(100*((iEvent+1)/len(self.allEvents))))

            # *** Load event lists
            _tracks_phi = [ track_phi for track_phi in self.tracks[iEvent][0] ]
            _tracks_rap = [ track_rap for track_rap in self.tracks[iEvent][1] ]
            _tracks_pt  = [ track_pt for track_pt in self.tracks[iEvent][2] ]
            _tracks_tlv = [ track_tlv for track_tlv in self.tracks[iEvent][3] ]

            _nHadrons_phi = [ nHadron_phi for nHadron_phi in self.nHadrons[iEvent][0] ]
            _nHadrons_rap = [ nHadron_rap for nHadron_rap in self.nHadrons[iEvent][1] ]
            _nHadrons_pt  = [ nHadron_pt for nHadron_pt in self.nHadrons[iEvent][2] ]
            _nHadrons_tlv = [ nHadron_tlv for nHadron_tlv in self.nHadrons[iEvent][3] ]
            
            _photons_phi = [ photon_phi for photon_phi in self.photons[iEvent][0] ]
            _photons_rap = [ photon_rap for photon_rap in self.photons[iEvent][1] ]
            _photons_pt  = [ photon_pt for photon_pt in self.photons[iEvent][2] ]
            _photons_tlv = [ photon_tlv for photon_tlv in self.photons[iEvent][3] ]
            
            # *** Handle pT/ET weights
            _tracks_weights   = [ 1/sum(_tracks_pt)   * x for x in _tracks_pt ]
            _nHadrons_weights = [ 1/sum(_nHadrons_pt)   * x for x in _nHadrons_pt ]
            _photons_weights   = [ 1/sum(_photons_pt)   * x for x in _photons_pt ]
            
            _tracks_sumWeights   = sum( _tracks_weights )
            _nHadrons_sumWeights = sum( _nHadrons_weights )
            _photons_sumWeights  = sum( _photons_weights )
            
            _totalWeights = _tracks_sumWeights + _photons_sumWeights + _nHadrons_sumWeights
            
            # *** Make total 4-vector for CM (center-of-mass)
            v_all         = TLorentzVector.PtEtaPhiMassLorentzVector(0,0,0,0)
            for tlv in _tracks_tlv:
                v_all += tlv
            for tlv in _nHadrons_tlv:
                v_all += tlv
            for tlv in _photons_tlv:
                v_all += tlv
            #print("pt: {}, eta: {}, phi: {}, E: {}".format(v_all.pt, v_all.eta, v_all.phi, v_all.E))

            # *** Initial CMS calculations
            _tracks_scaled_phi   = [ x*y for x,y in zip(_tracks_phi, _tracks_weights)]
            _nHadrons_scaled_phi = [ x*y for x,y in zip(_nHadrons_phi, _nHadrons_weights)]
            _photons_scaled_phi  = [ x*y for x,y in zip(_photons_phi, _photons_weights)]
            _tracks_scaled_rap   = [ x*y for x,y in zip(_tracks_rap, _tracks_weights)]
            _nHadrons_scaled_rap = [ x*y for x,y in zip(_nHadrons_rap, _nHadrons_weights)]
            _photons_scaled_rap  = [ x*y for x,y in zip(_photons_rap, _photons_weights)]
            _phi_centroid = sum( _tracks_scaled_phi + _nHadrons_scaled_phi + _photons_scaled_phi ) / _totalWeights
            _rap_centroid = sum( _tracks_scaled_rap + _nHadrons_scaled_rap + _photons_scaled_rap ) / _totalWeights
            #_phi_centroid = v_all.phi
            #print("nJets: {}, phi centroid: {}, rap centroid: {}".format(event['nJets'], _phi_centroid, _rap_centroid))
            
            # *** Phi Rotation
            _tracks_phi_rot   = [self.rotateByPhi(x, _phi_centroid) for x in _tracks_phi ]
            _nHadrons_phi_rot = [self.rotateByPhi(x, _phi_centroid) for x in _nHadrons_phi ]
            _photons_phi_rot  = [self.rotateByPhi(x, _phi_centroid) for x in _photons_phi ]
            
            _tracks_scaledRotated_phi   = [ x*y for x,y in zip(_tracks_phi_rot, _tracks_weights)]
            _nHadrons_scaledRotated_phi = [ x*y for x,y in zip(_nHadrons_phi_rot, _nHadrons_weights)]
            _photons_scaledRotated_phi  = [ x*y for x,y in zip(_photons_phi_rot, _photons_weights)]
            _phi_centroid_CM = sum( _tracks_scaledRotated_phi + _nHadrons_scaledRotated_phi + _photons_scaledRotated_phi ) / _totalWeights
            
            # *** Rapidity Boost
            # Method A: y_particle,lab = y_particle,CM + y_CM,lab
            _rap_cm = v_all.rapidity
            _tracks_rap_boost   = [ jc_rap - _rap_cm for jc_rap in _tracks_rap]
            _nHadrons_rap_boost = [ jc_rap - _rap_cm for jc_rap in _nHadrons_rap]
            _photons_rap_boost  = [ jc_rap - _rap_cm for jc_rap in _photons_rap]

            ## Method B: Use 4vector math and lorentz boosts
            #tlvz  = TLorentzVector.TLorentzVector(0, 0, -1*v_all.p3.z, v_all.t)
            #boost_z   = (tlvz.p3/tlvz.p3.mag)*tlvz.beta
            #_tracks_rap_boost2   = [ tlv.boost( boost_z).rapidity for tlv in _tracks[3]]
            #_nHadrons_rap_boost2 = [ tlv.boost( boost_z).rapidity for tlv in _nHadrons[3]]
            #_photons_rap_boost2  = [ tlv.boost( boost_z).rapidity for tlv in _photons[3]]
            #_tracks_pt_boost   = [ tlv.boost( boost_z).pt for tlv in _tracks[3]]
            #_nHadrons_pt_boost = [ tlv.boost( boost_z).pt for tlv in _nHadrons[3]]
            #_photons_pt_boost  = [ tlv.boost( boost_z).pt for tlv in _photons[3]]
            
            #for i in range(0, len(_tracks_pt_boost)):
            #    if (_tracks_pt_boost[i] - _tracks[2][i]) > 0.001:
            #        print("shit, {} {}".format(_tracks_pt_boost[i] , _tracks[2][i]))
            #for i in range(0, len(_tracks_rap_boost)):
            #    if (_tracks_rap_boost[i] - _tracks_rap_boost2[i]) > 0.001:
            #        print("shit2, {} {}".format(_tracks_rap_boost[i] , _tracks_rap_boost2[i]))

            _tracks_scaledRotated_rap   = [ x*y for x,y in zip(_tracks_rap_boost, _tracks_weights)]
            _nHadrons_scaledRotated_rap = [ x*y for x,y in zip(_nHadrons_rap_boost, _nHadrons_weights)]
            _photons_scaledRotated_rap  = [ x*y for x,y in zip(_photons_rap_boost, _photons_weights)]
        
            _rap_centroid_CM = sum( _tracks_scaledRotated_rap + _nHadrons_scaledRotated_rap + _photons_scaledRotated_rap ) / _totalWeights
        
            #print("nJets: {}, phi centroid (CM): {}, rap centroid (CM): {}".format(event['nJets'], _phi_centroid_CM, _rap_centroid_CM))

            # *** Make aggregates of all events
            self.final_tracks[0].append( _tracks_phi_rot)
            self.final_tracks[1].append( _tracks_rap_boost)
            self.final_tracks[2].append( _tracks_weights)
            _final_tracks[0] += _tracks_phi_rot
            _final_tracks[1] += _tracks_rap_boost
            _final_tracks[2] += _tracks_weights

            self.final_nHadrons[0].append( _nHadrons_phi_rot)
            self.final_nHadrons[1].append( _nHadrons_rap_boost)
            self.final_nHadrons[2].append( _nHadrons_weights)
            _final_nHadrons[0] += _nHadrons_phi_rot
            _final_nHadrons[1] += _nHadrons_rap_boost
            _final_nHadrons[2] += _nHadrons_weights

            self.final_photons[0].append( _photons_phi_rot)
            self.final_photons[1].append( _photons_rap_boost)
            self.final_photons[2].append( _photons_weights)
            _final_photons[0] += _photons_phi_rot
            _final_photons[1] += _photons_rap_boost
            _final_photons[2] += _photons_weights


            # *** Make a few single-event image plots for eye-tests
            if iEvent < 3: 
                # Raw-raw, i.e. no pT weighting
                self.makeSimplePlots( [_tracks_phi, _tracks_rap, _tracks_pt], 'Tracks', False, 'SingleEvent{}'.format(iEvent) )
                self.makeSimplePlots( [_nHadrons_phi, _nHadrons_rap, _nHadrons_pt], 'Neutral Hadrons', False, 'SingleEvent{}'.format(iEvent) )
                self.makeSimplePlots( [_photons_phi, _photons_rap, _photons_pt], 'Photons', False, 'SingleEvent{}'.format(iEvent) )
                
                # Semi-raw, i.e. no boost/rotation but using pT weighting
                self.makeSimplePlots( [_tracks_phi, _tracks_rap, _tracks_pt], 'Tracks', True, 'SingleEvent{}'.format(iEvent) )
                self.makeSimplePlots( [_nHadrons_phi, _nHadrons_rap, _nHadrons_pt], 'Neutral Hadrons', True, 'SingleEvent{}'.format(iEvent) )
                self.makeSimplePlots( [_photons_phi, _photons_rap, _photons_pt], 'Photons', True, 'SingleEvent{}'.format(iEvent) )

                # Semi-processed, i.e. only rotation and with pT weighting
                self.makeSimplePlots( [_tracks_phi_rot, _tracks_rap, _tracks_pt], 'Tracks', True, 'SingleEvent{}_phiRotation'.format(iEvent) )
                self.makeSimplePlots( [_nHadrons_phi_rot, _nHadrons_rap, _nHadrons_pt], 'Neutral Hadrons', True, 'SingleEvent{}_phiRotation'.format(iEvent) )
                self.makeSimplePlots( [_photons_phi_rot, _photons_rap, _photons_pt], 'Photons', True, 'SingleEvent{}_phiRotation'.format(iEvent) )

                # Processed, i.e. rotation+boost and with pT weighting
                self.makeSimplePlots( [_tracks_phi_rot, _tracks_rap_boost, _tracks_pt], 'Tracks', True, 'SingleEvent{}_phiRotationRapBoost'.format(iEvent) )
                self.makeSimplePlots( [_nHadrons_phi_rot, _nHadrons_rap_boost, _nHadrons_pt], 'Neutral Hadrons', True, 'SingleEvent{}_phiRotationRapBoost'.format(iEvent) )
                self.makeSimplePlots( [_photons_phi_rot, _photons_rap_boost, _photons_pt], 'Photons', True, 'SingleEvent{}_phiRotationRapBoost'.format(iEvent) )


        # *** Make some globally averaged plots after processing
        self.makeSimplePlots( _final_tracks, 'Tracks', True, 'Final' )
        self.makeSimplePlots( _final_nHadrons, 'Neutral Hadrons', True, 'Final' )
        self.makeSimplePlots( _final_photons, 'Photons', True, 'Final')
                
        return
    

    def makeEventImages(self):
        """ construct the event images """

        # *** Set options for saving (bins, range, etc)
        _imgOpts = dict( _nbins_phi=self.pixelWidth, _range_phi=[-1*np.pi-0.5, np.pi+0.5], _nbins_rap=self.pixelWidth, _range_rap=[-3.0, 3.0] )
        _compositeImages = []
        _compositeImages_4j = []
        for iEvent in range(0, len(self.final_tracks[0])):

            # *** Loosely keep track of events
            if 100*((iEvent+1)/len(self.allEvents))%10 == 0:
                print('{}% Imaged'.format(100*((iEvent+1)/len(self.allEvents))))

            # *** Make event images
            _tracks_img = self.function_hist2d( self.final_tracks[0][iEvent], self.final_tracks[1][iEvent], self.final_tracks[2][iEvent], **_imgOpts)
            _nHadrons_img = self.function_hist2d( self.final_nHadrons[0][iEvent], self.final_nHadrons[1][iEvent], self.final_nHadrons[2][iEvent], **_imgOpts)
            _photons_img = self.function_hist2d( self.final_photons[0][iEvent], self.final_photons[1][iEvent], self.final_photons[2][iEvent], **_imgOpts)
            _composite_img = np.stack( (_tracks_img, _nHadrons_img, _photons_img), axis = -1)
            
            self.final_tracks[3].append( _tracks_img )
            self.final_nHadrons[3].append( _nHadrons_img )
            self.final_photons[3].append( _photons_img )

            # *** Make composite image (15, 15, 3)
            _compositeImages.append( _composite_img )
            if self.nJets[iEvent] >= 4:
                _compositeImages_4j.append( _composite_img )

        # *** Append to global list
        self.final_images[0] += self.final_tracks[3]
        self.final_images[1] += self.final_nHadrons[3]
        self.final_images[2] += self.final_photons[3]
        self.final_images[3] += _compositeImages
        self.final_images[4] += _compositeImages_4j
        self.final_eventQuantities[0] += self.nJets
        self.final_eventQuantities[1] += self.nBTags
        
        return
    

    def setPhiRapidityPtLists(self):
        """ return three lists of phi, rapidity, pt for later plotting """

        for iEvent in range(0, len(self.allEvents)):

            # *** Test run only
            if iEvent > 3 and self.isTestRun==True:
                break
            
            # *** Loosely keep track of events
            event = self.allEvents[ iEvent ]
            if 100*((iEvent+1)/len(self.allEvents))%25 == 0:
                print('{}% Loaded'.format(100*((iEvent+1)/len(self.allEvents))))

            # *** Protection against events with 0 jets
            if event['nJets'] == 0:
            #if event['nJets'] < 2:
                continue

            # *** Load constituent collections
            self.tracks.append( self.returnSingleEventPhiRapidityPtList( event, 'Tracks') )
            self.nHadrons.append( self.returnSingleEventPhiRapidityPtList( event, 'Neutral Hadrons'))
            self.photons.append( self.returnSingleEventPhiRapidityPtList( event, 'Photons'))

            # *** 3. Set event-level quantities
            self.nJets.append( event['nJets'] )
            self.nBTags.append( event['nBTags'] )

        return

    
    def returnPlotOpts(self, _consLabel ):
        """ common function for returning plotting opts"""

        track_plotOpts  = dict(bins=(self.pixelWidth, self.pixelWidth), range=[[-1*np.pi-0.5, np.pi+0.5],[-3.0, 3.0]], cmap=plt.cm.Reds)
        photon_plotOpts = dict(bins=(self.pixelWidth, self.pixelWidth), range=[[-1*np.pi-0.5, np.pi+0.5],[-3.0, 3.0]], cmap=plt.cm.Blues)
        nHad_plotOpts   = dict(bins=(self.pixelWidth, self.pixelWidth), range=[[-1*np.pi-0.5, np.pi+0.5],[-3.0, 3.0]], cmap=plt.cm.Greens)

        _plotOpts = {}
        if _consLabel == 'Tracks':
            _plotOpts = track_plotOpts
        elif _consLabel == 'Photons':
            _plotOpts = photon_plotOpts
        elif _consLabel == 'Neutral Hadrons':
            _plotOpts = nHad_plotOpts

        return _plotOpts

    def makeSimplePlots(self, constituentList, consLabel='', _ptWeighted=False, stageLabel='Raw', nEventsToAverage=-1):
        """ make plots"""

        _plotDir = self.datasetName+'/plots/'
        if os.path.isdir( _plotDir )==False:
            os.mkdir( _plotDir )

        _plotOpts = self.returnPlotOpts( consLabel )
        _sampleLabel = 'Dihiggs' if self.isSignal else 'QCD'
        _weightLabel = 'Pt-Weighted' if _ptWeighted else 'Unweighted'
        _plotTitle   = '{} {} {} ({})'.format(stageLabel, _weightLabel, consLabel, _sampleLabel)
        
        _phi = constituentList[0]
        _rap = constituentList[1]
        _pt  = constituentList[2]
        
        if nEventsToAverage>0:
            _pt = [ constituent/nEventsToAverage for constituent in _pt ]

        if _ptWeighted:
            bins, x_edges, y_edges, img = plt.hist2d( _phi, _rap, weights=_pt, **_plotOpts )
        else:
            bins, x_edges, y_edges, img = plt.hist2d( _phi, _rap, **_plotOpts )

        bin_vals = [v for row in bins for v in row]
        #plt.hist(bin_vals)
        #plt.show()
        b_mean = np.mean(bin_vals)
        b_std = np.std(bin_vals)
        plt.clim( 0, b_mean + b_std)
        
        plt.title( _plotTitle )
        plt.xlabel('Phi')
        plt.ylabel('Rapidity')

        plt.savefig('{}/{}_{}_{}_{}.png'.format( _plotDir, consLabel, _sampleLabel, stageLabel, _weightLabel).replace(' ',''))
        
        return

    def returnSingleEventPhiRapidityPtList(self, _event, _consLabel ):
        """ return single list of phi, rapidity, pt"""
    
        # *** 1. Set some collection-dependent variables
        _consCode = -1
        if _consLabel == 'Tracks':
            _consCode = 0
        elif _consLabel == 'Neutral Hadrons':
            _consCode = 1
        elif _consLabel == 'Photons':
            _consCode = 2
    
        # *** 1. Get the phi/pt/rapidity from the stored constituents
        _rap = [ constituent[5] for constituent in _event['Constituents'] if constituent[6]==_consCode ]
        _phi = [ constituent[3] for constituent in _event['Constituents'] if constituent[6]==_consCode ]
        _pt  = [ constituent[1] for constituent in _event['Constituents'] if constituent[6]==_consCode ]
        
        # *** 2. Get TLorentzVectors
        _tlv = [JetCons(jc[0], jc[1], jc[2], jc[3], jc[4]).cons_LVec for jc in _event['Constituents']  if jc[6]==_consCode ]
        #v_all = TLorentzVector.PtEtaPhiMassLorentzVector(0,0,0,0)
        #for tlv in allTLVs:
        #    v_all += tlv
        #print("pt: {}, eta: {}, phi: {}, E: {}".format(v_all.pt, v_all.eta, v_all.phi, v_all.E))
        
        _all = [_phi, _rap, _pt, _tlv]
        
        return _all
    
    def rotateByPhi(self, _phi, _rotAngle):
        return (((_phi - _rotAngle)+np.pi) % (2*np.pi))-np.pi 


    def function_hist2d(self, _phi, _rap, _pt, _nbins_phi, _range_phi, _nbins_rap, _range_rap):

        # make bins
        _bins_phi = np.linspace( _range_phi[0], _range_phi[1], _nbins_phi+1)
        _bins_rap = np.linspace( _range_rap[0], _range_rap[1], _nbins_rap+1)
        #weights_phi = np.ones_like(a)/float(len(a))
        
        H, xedges, yedges = np.histogram2d( _rap, _phi, bins=(_bins_rap, _bins_phi), weights=_pt)
        
        return H

    def saveFilesToH5(self):
        """ save all image files in hdf5 format"""

        _imgDir = self.datasetName+'/images/'
        if os.path.isdir( _imgDir )==False:
            os.mkdir( _imgDir )

        if len(self.final_images[0]) != len(self.final_eventQuantities[0]):
            print("event quantity mismatch with number of images!!! imgs:{} quantities:{}".format(len(self.final_images[0]) , len(self.final_eventQuantities[0])))
                
        hf = h5.File( '{}/{}_allImages.h5'.format(_imgDir, self.datasetName), 'w')
        hf.create_dataset('trackImgs', data=self.final_images[0], compression="gzip", compression_opts=3)
        hf.create_dataset('nHadronImgs', data=self.final_images[1], compression="gzip", compression_opts=3)
        hf.create_dataset('photonImgs', data=self.final_images[2], compression="gzip", compression_opts=3)
        hf.create_dataset('compositeImgs', data=self.final_images[3], compression="gzip", compression_opts=3)
        hf.create_dataset('compositeImgs_4j', data=self.final_images[3], compression="gzip", compression_opts=3)
        hf.create_dataset('nJets', data = self.final_eventQuantities[0], compression="gzip", compression_opts=3)
        hf.create_dataset('nBTags', data = self.final_eventQuantities[1], compression="gzip", compression_opts=3)

        hf.close()

        return
