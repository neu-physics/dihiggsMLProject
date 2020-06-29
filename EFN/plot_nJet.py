import pickle
import numpy as np
import argparse
import subprocess

import matplotlib.pyplot as plt
import energyflow as ef
from energyflow.archs import EFN
from energyflow.utils import data_split, to_categorical
from sklearn.metrics import roc_auc_score, roc_curve
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf

import sys
#sys.path.insert(0, '/afs/cern.ch/work/l/lian/public/diHiggs/dihiggsMLProject/')
sys.path.insert(0, '/uscms/home/ali/nobackup/diHiggs/dihiggsMLProject/')
from utils.commonFunctions import *

parser = argparse.ArgumentParser()
parser.add_argument('--sigInput',dest='sigInput')
parser.add_argument('--bkgInput',dest='bkgInput')
parser.add_argument('--nSig',dest='nSig',type=int, default=-1)
parser.add_argument('--nBkg',dest='nBkg',type=int, default=-1)
parser.add_argument('--nJets',dest='nJets', type=int, default=1)

args = parser.parse_args()

Jet_sig = []
nsig = 0
nbkg = 0

with open(args.sigInput) as sig_path:
  line = sig_path.readline()
  while (line and args.nSig>=0 and nsig<args.nSig ):
    f_path = line.strip('\n')
    f_sig = open(f_path,'rb')
    sig = pickle.load(f_sig)
    f_sig.close()

    Jet_sig.extend(sig)
    line = sig_path.readline()
    nsig += len(sig)

nJet_sig = []
for evt in Jet_sig:
  nJet_sig.append(evt['nJets'])

Jet_bkg = []
with open(args.bkgInput) as bkg_path:
  line = bkg_path.readline()
  while (line and args.nBkg>=0 and nbkg<args.nBkg ):
    f_path = line.strip('\n')
    f_bkg = open(f_path,'rb')
    bkg = pickle.load(f_bkg)
    f_bkg.close()

    Jet_bkg.extend(bkg)
    line = bkg_path.readline()
    nbkg += len(bkg)

nJet_bkg = []
for evt in Jet_bkg:
  nJet_bkg.append(evt['nJets'])
plt.hist(nJet_sig, bins=30, alpha=0.5, density=True, label="signal")
plt.hist(nJet_bkg, bins=30, alpha=0.5, density=True, label="background")
plt.legend(loc="best")
plt.title("Number of Jets")
plt.savefig("nJet.png")

print("Finished loading data :)")

