import pickle
import numpy as np
import argparse
import subprocess

import matplotlib.pyplot as plt
import energyflow as ef
from energyflow.archs import EFN
from energyflow.utils import data_split, to_categorical
from sklearn.metrics import roc_auc_score, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument('--sigInput',dest='sigInput')
parser.add_argument('--bkgInput',dest='bkgInput')

args = parser.parse_args()

def preprocess(x):
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[mask,0].sum()
    return x

def MakeSameLengthForGivenIdx(data,y,n,idx=[1,3,5]):   #data is raw data, n is the number of Jet constituents to keep, d is the dimension of each jet constituents(the number of variables in data)
    data_modified_list = []
    valid_idx = []
    for ievt, x in enumerate(data):
        if(len(x)==0): 
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

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    return plt

def plot_EFNScore(Y_test, preds):
    sig_mask = Y_test[:,1]==1
    bkg_mask = Y_test[:,1]==0
    sig_preds = preds[sig_mask,1]
    bkg_preds = preds[bkg_mask,1]
    plt.hist(sig_preds, bins=50, alpha=0.5, density=True, label="signal")
    plt.hist(bkg_preds, bins=50, alpha=0.5, density=True, label="background")
    plt.legend(loc="best")
    plt.title("EFN score")
    return plt

Jet_cons = []
y_raw = []
with open(args.sigInput) as sig_path:
  line = sig_path.readline()
  while line:
    f_path = line.strip('\n')
    f_sig = open(f_path,'rb')
    sig = pickle.load(f_sig)
    f_sig.close()

    Jet_cons.extend(sig)
    y_sig = np.ones(len(sig))
    y_raw.extend(y_sig)
    line = sig_path.readline()
    #break

with open(args.bkgInput) as bkg_path:
  line = bkg_path.readline()
  while line:
    f_path = line.strip('\n')
    f_bkg = open(f_path,'rb')
    bkg = pickle.load(f_bkg)
    f_bkg.close()

    Jet_cons.extend(bkg)
    y_bkg = np.zeros(len(bkg))
    y_raw.extend(y_bkg)
    line = bkg_path.readline()
    #break

print("Finished loading data :)")

n = int(200)
data,y = MakeSameLengthForGivenIdx(Jet_cons,y_raw,n)
X_list = []
print("SameLength finished :)")

for evt in data:
    if(len(evt)==0):
        continue
    x = preprocess(evt)
    X_list.append(x)
    
X = np.array(X_list)
Y = to_categorical(y, num_classes=2)
print("preprocess finished :)")

# network training parameters
batch_size = 500
val = 0.2
test = 0.2
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
num_epoch = 20
(z_train, z_val, z_test, 
 p_train, p_val, p_test,
 Y_train, Y_val, Y_test) = data_split(X[:,:,0], X[:,:,1:], Y, val=val, test=test)

efn = EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes)
history = efn.fit([z_train, p_train], Y_train,
          epochs=num_epoch,
          batch_size=batch_size,
          validation_data=([z_val, p_val], Y_val),
          verbose=1)


preds = efn.predict([z_test, p_test], batch_size=1000)
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print('EFN AUC:', auc)
p_history = plot_history(history)
p_history.savefig("lossHistory.png")
p_history.close()
p_score = plot_EFNScore(Y_test,preds)
p_score.savefig("score.png")
p_score.close()
print("Plots saved :)")




