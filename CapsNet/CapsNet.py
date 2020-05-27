# Reference : https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
import h5py as h5
import sys
#sys.path.insert(0, '/afs/cern.ch/work/l/lian/public/diHiggs/dihiggsMLProject/')
#sys.path.insert(0, '/Users/flywire/Desktop/sci/diHiggs/CapsNet')

sys.path.insert(0, '/uscms/home/ali/nobackup/diHiggs/dihiggsMLProject/')
from CapsLayers import CapsuleLayer, PrimaryCaps, Length, Mask
from utils.commonFunctions import *

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, n_routings, decoder=False):
    
    # input layer
    x = layers.Input(shape=input_shape)
    y = layers.Input(shape=(n_class,))
    
    # conv layer 1
    conv1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)
    
    # layer 2: Primary Caps Layer
    primarycaps = PrimaryCaps(conv1, dim_capsule=8, n_channels=32, kernel_size=3, strides=2, padding='valid')
    
    # layer 3: Digits Caps Layer
    digitcaps_1 = CapsuleLayer(n_capsule=10, dim_capsule=16, n_routings=n_routings, name='digitCaps_1')(primarycaps)
    #digitcaps_2 = CapsuleLayer(n_capsule=4, dim_capsule=12, n_routings=n_routings, name='digitCaps_2')(digitcaps_1)
    digitcaps = CapsuleLayer(n_capsule=n_class, dim_capsule=8, n_routings=n_routings, name='digitCaps')(digitcaps_1)
    
    # layer 4: length layer 
    out_caps = Length(name='capsnet')(digitcaps)
    
    #model = models.Model(inputs = x, outputs = out_caps, name='CapsNet')
    
    if(decoder):
        mask_by_y = Mask()([digitcaps, y])  # Use true label to mask the digitCaps layer for training
        mask = Mask()(digitcaps)  # Use length of calsules to mask for prediction
        
        decoder = models.Sequential(name='decoder')
        #Fully connected network to reconstruct the image from the active capsule
        decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
        decoder.add(layers.Dense(1024, activation='relu'))
        decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=input_shape, name='out_reco'))
        
        train_model = models.Model([x,y], [out_caps, decoder(mask_by_y)])
        eval_model = models.Model(x, [out_caps, decoder(mask)])
        
    else:
        train_model = models.Model(inputs = x, outputs = out_caps, name='CapsNet')
        eval_model = models.Model(inputs = x, outputs = out_caps, name='CapsNet')
    
    return train_model, eval_model

def load_data(SigPath, BkgPath, nSig=-1, nBkg=-1):
  """
  load data:
  filename: txt file that stores the path of all data files
  isSignal: whether the data is signal or bkg
  nEvents: the number of events to use
  """
  nSig_loaded = 0
  nBkg_loaded = 0
  dataset_sig = []
  dataset_bkg = []
  with open(SigPath) as path:
    line = path.readline()
    while(line and (( nSig>=0 and nSig_loaded<nSig) or (nSig==-1))):
      f_path = line.strip('\n')
      f = h5.File(f_path, 'r')
      data = f['events']['compositeImages']
      if(len(dataset_sig)==0):
        dataset_sig = data
      else:
        dataset_sig = np.concatenate( (dataset_sig, data))
      nSig_loaded += data.shape[0]
      f.close()
      line = path.readline()
    y_sig = np.ones((dataset_sig.shape[0],))

  with open(BkgPath) as path:
    line = path.readline()
    while(line and (( nBkg>=0 and nBkg_loaded<nBkg) or (nBkg==-1))):
      f_path = line.strip('\n')
      f = h5.File(f_path, 'r')
      data = f['events']['compositeImages']
      if(len(dataset_bkg)==0):
        dataset_bkg = data
      else:
        dataset_bkg = np.concatenate( (dataset_bkg, data))
      nBkg_loaded += data.shape[0]
      f.close()
      line = path.readline()
    y_bkg = np.zeros((dataset_bkg.shape[0],))

  dataset_total = np.concatenate((dataset_sig, dataset_bkg))
  y_total = np.concatenate((y_sig, y_bkg))
  y_total = to_categorical(y_total)
  X, y = shuffle(dataset_total, y_total)
  #X_train, X_test, y_train, y_test = train_test_split(dataset_total, y_total, test_size=0.2, shuffle=True)
  
  print("Finished loading data :)")
  print("# sig: {0}  # bkg: {1} # events: {2}".format(nSig_loaded,nBkg_loaded,dataset_total.shape[0]))
  return (X, y), nSig_loaded, nBkg_loaded



def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    #x_train = x_train[0:200]
    #x_test = x_test[0:10]
    #y_train = y_train[0:200]
    #y_test = y_test[0:10]
    return (x_train, y_train), (x_test, y_test)

def train(model, data, args):
    name = ''
    (x_train, y_train) = data

    if(not args.decoder):
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tb-logs', histogram_freq=int(args.debug))
        es = callbacks.EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=10, restore_best_weights=True)
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights.h5', monitor='val_auc',
                                               mode='max', save_best_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[margin_loss],
                      loss_weights=[1.],
                      metrics={'capsnet': ['accuracy', auc]}
                     )
    
        history = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, 
                  validation_split=0.2, callbacks=[log, tb, es, checkpoint, lr_decay])
    
    else:
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tb-logs', histogram_freq=int(args.debug))
        es = callbacks.EarlyStopping(monitor='val_capsnet_auc', mode='max', verbose=1, patience=20, restore_best_weights=True)
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights.h5', monitor='val_capsnet_auc',
                                               mode='max', save_best_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., args.lam_reco],
                      metrics={'capsnet': ['accuracy', auc]}
                     )
        
        history = model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs, 
                  validation_split=0.2, callbacks=[log, tb, es, checkpoint, lr_decay])
    
    model.save_weights(args.save_dir + '/trained_model.h5')
    print("Trained Model saved to {0}/{1}".format(args.save_dir, 'trained_model.h5'))
    plot_history(history,'loss',name,args.save_dir)
    if(args.decoder):
      plot_history(history,'capsnet_accuracy',name,args.save_dir)
      plot_history(history,'capsnet_auc',name,args.save_dir)
    else:
      plot_history(history,'accuracy',name,args.save_dir)
      plot_history(history,'auc',name,args.save_dir)
    
    return model



def test(model, data, nSigTest, nBkgTest, args):
    x_test, y_test = data
    name = ''
    if (not args.decoder):
        y_preds = model.predict(x=x_test)
    else:
        y_preds, x_reco = model.predict(x_test)
    print('Test acc: {}'.format(np.sum(np.argmax(y_preds, 1) == np.argmax(y_test, 1))/y_test.shape[0]))
    plot_ROC(y_test,y_preds,name,args.save_dir)
    plot_MLScore(y_test,y_preds,nSigTest,nBkgTest,name,args.save_dir)
    return 
    

def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lmbd=0.5):
    """
    Margin Loss function given in CapsNet paper
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L,1))

    
def auc( y_true, y_pred ) :
        score = tf.py_function( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                                [y_true, y_pred],
                                'float32',
                                name='sklearnAUC' )
        return score

def plot_history(history, tag='loss', name='', savedir='./'):
    plt.plot(history.history[tag])
    plt.plot(history.history["val_{}".format(tag)])
    plt.title('model_{}'.format(tag))
    plt.ylabel(tag)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(savedir + '/history_{0}_{1}.png'.format(tag,name))
    plt.close()
    return 

def plot_MLScore(Y_test, preds, nsigTest, nbkgTest, name='', savedir='./'):
    sig_mask = Y_test[:,1]==1
    bkg_mask = Y_test[:,1]==0
    sig_preds = preds[sig_mask,1]
    bkg_preds = preds[bkg_mask,1]
    plt.hist(sig_preds, bins=30, alpha=0.5, density=True, stacked=True, label="signal")
    plt.hist(bkg_preds, bins=30, alpha=0.5, density=True, stacked=True, label="background")
    plt.legend(loc="best")
    plt.title("CapsNet score")
    plt.xlabel('score')
    plt.ylabel('A.U.')
    returnBestCutValue('CapsNet',sig_preds.copy(), bkg_preds.copy(), hh_nEventsGen = nsigTest, qcd_nEventsGen = nbkgTest)
    plt.savefig(savedir + "/score_{}.png".format(name))
    plt.close()
    return 

def plot_ROC(Y_test, preds, name='', savedir='./'):
    fpr, tpr, _ = roc_curve(Y_test[:,1], preds[:,1])
    plt.plot(fpr, tpr, label=name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    rocCurve = np.array([fpr,tpr])
    np.savetxt(savedir + '/ROC_curve_{}.txt'.format(name),rocCurve)
    plt.savefig(savedir + "/ROC_{}.png".format(name))
    plt.close()
    return 

