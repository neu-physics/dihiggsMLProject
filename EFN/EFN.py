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

def auc( y_true, y_pred ) :
      score = tf.py_function( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                          [y_true, y_pred],
                          'float32',
                          name='sklearnAUC' )
      return score

def preprocess(x):
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[mask,0].sum()
    return x

def MakeSameLengthForGivenIdx(data,y,n,idx=[1,3,5],nJets=1):   #data is raw data, n is the number of Jet constituents to keep, idx is the index of useful variables of each jet constituents(the number of variables in data)
    data_modified_list = []
    valid_idx = []
    for ievt, x_dict in enumerate(data):
        x = x_dict['Constituents']
        if(x_dict['nJets']<nJets): 
        #    data_modified_list.append(np.zeros((n,d)))
            continue
        elif(len(x)>=n):
            data_modified_list.append(x[:n,idx])
        else:
            idx_mod = n-len(x)
            pad = np.zeros((idx_mod,len(idx)))
            data_modified_list.append(np.concatenate((x[:,idx],pad)))
        valid_idx.append(ievt)
    data_mod = np.array(data_modified_list)
    y_mod = np.array(y)[valid_idx]
    return data_mod,y_mod

def plot_history(history,tag='loss'):
    plt.plot(history.history[tag])
    plt.plot(history.history["val_{}".format(tag)])
    plt.title('model_{}'.format(tag))
    plt.ylabel(tag)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    return plt

def plot_EFNScore(Y_test, preds, nsig, nbkg):
    sig_mask = Y_test[:,1]==1
    bkg_mask = Y_test[:,1]==0
    sig_preds = preds[sig_mask,1]
    bkg_preds = preds[bkg_mask,1]
    plt.hist(sig_preds, bins=30, alpha=0.5, density=True, label="signal")
    plt.hist(bkg_preds, bins=30, alpha=0.5, density=True, label="background")
    plt.legend(loc="best")
    plt.title("EFN score")
    returnBestCutValue('EFN',sig_preds.copy(), bkg_preds.copy(), _testingFraction=0.2, hh_nEventsGen = nsig, qcd_nEventsGen = nbkg)
    return plt

Jet_cons = []
y_raw = []
nsig = 0
nbkg = 0

with open(args.sigInput) as sig_path:
  line = sig_path.readline()
  while (line and args.nSig>=0 and nsig<args.nSig ):
    f_path = line.strip('\n')
    f_sig = open(f_path,'rb')
    sig = pickle.load(f_sig)
    f_sig.close()

    Jet_cons.extend(sig)
    y_sig = np.ones(len(sig))
    y_raw.extend(y_sig)
    line = sig_path.readline()
    nsig += len(sig)

with open(args.bkgInput) as bkg_path:
  line = bkg_path.readline()
  while (line and args.nBkg>=0 and nbkg<args.nBkg ):
    f_path = line.strip('\n')
    f_bkg = open(f_path,'rb')
    bkg = pickle.load(f_bkg)
    f_bkg.close()

    Jet_cons.extend(bkg)
    y_bkg = np.zeros(len(bkg))
    y_raw.extend(y_bkg)
    line = bkg_path.readline()
    nbkg += len(bkg)

print("Finished loading data :)")
print("# sig: {0}  # bkg: {1} # events: {2}".format(nsig,nbkg,len(Jet_cons)))
n = int(200)
data,y = MakeSameLengthForGivenIdx(Jet_cons,y_raw,n,nJets = args.nJets)
X_list = []

for evt in data:
    if(len(evt)==0):
        continue
    x = preprocess(evt)
    X_list.append(x)
    
X = np.array(X_list)
Y = to_categorical(y, num_classes=2)

# network training parameters
batch_size = 500
val = 0.2
test = 0.2
Phi_sizes, F_sizes = (200, 200, 256), (200, 200, 200)
num_epoch = 1000
(z_train, z_val, z_test, 
 p_train, p_val, p_test,
 Y_train, Y_val, Y_test) = data_split(X[:,:,0], X[:,:,1:], Y, val=val, test=test)
es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=50, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', monitor='val_auc', mode='max', verbose=1, save_best_only=True)
efn = EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes, metrics=['acc', auc], Phi_l2_regs=5e-05, F_l2_regs=5e-05)
history = efn.fit([z_train, p_train], Y_train,
          epochs=num_epoch,
          batch_size=batch_size,
          validation_data=([z_val, p_val], Y_val),
          verbose=1, callbacks=[es, mc])

dependencies = {
  'auc': tf.keras.metrics.AUC(name="auc")
}
saved_model = load_model('best_model.h5', custom_objects=dependencies)
preds = saved_model.predict([z_test, p_test], batch_size=1000)
#preds = efn.predict([z_test, p_test], batch_size=1000)
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print('EFN AUC:', auc)

#save plots
p_loss = plot_history(history,'loss')
p_loss.savefig("lossHistory.png")
p_loss.close()
p_acc = plot_history(history,'acc')
p_acc.savefig("accHistory.png")
p_acc.close()
p_auc = plot_history(history,'auc')
p_auc.savefig("aucHistory.png")
p_auc.close()


p_score = plot_EFNScore(Y_test,preds,nsig,nbkg)
p_score.savefig("score.png")
p_score.close()
print("Plots saved :)")




