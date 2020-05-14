# Reference : https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import sys
#sys.path.insert(0, '/afs/cern.ch/work/l/lian/public/diHiggs/dihiggsMLProject/')
sys.path.insert(0, '/Users/flywire/Desktop/sci/diHiggs/CapsNet')

from CapsLayers import CapsuleLayer, PrimaryCaps, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, n_routings, decoder=False):
    
    # input layer
    x = layers.Input(shape=input_shape)
    y = layers.Input(shape=(n_class,))
    
    # conv layer 1
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    
    # layer 2: Primary Caps Layer
    primarycaps = PrimaryCaps(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    
    # layer 3: Digits Caps Layer
    digitcaps = CapsuleLayer(n_capsule=n_class, dim_capsule=16, n_routings=n_routings, name='digitCaps')(primarycaps)
    
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
    (x_train, y_train), (x_test, y_test) = data
    

   
    if(not args.decoder):
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tb-logs', histogram_freq=int(args.debug))
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_accuracy',
                                               save_best_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[margin_loss],
                      loss_weights=[1.],
                      metrics={'capsnet': 'accuracy'}
                     )
    
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, 
                  validation_data=[x_test,y_test], callbacks=[log, tb, checkpoint, lr_decay])
    
    else:
        log = callbacks.CSVLogger(args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tb-logs', histogram_freq=int(args.debug))
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_accuracy',
                                               save_best_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., args.lam_reco],
                      metrics={'capsnet': 'accuracy'}
                     )
        
        model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs, 
                  validation_data=[[x_test,y_test],[y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    
    model.save_weights(args.save_dir + '/trained_model.h5')
    print("Trained Model saved to {0}/{1}".format(args.save_dir, 'trained_model.h5'))
    
    return model



def test(model, data, args):
    x_test, y_test = data
    if (not args.decoder):
        y_preds = model.predict(x=x_test)
    else:
        y_preds, x_reco = model.predict(x_test)
    print('Test acc: {}'.format(np.sum(np.argmax(y_preds, 1) == np.argmax(y_test, 1))/y_test.shape[0]))
    return 
    

def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lmbd=0.5):
    """
    Margin Loss function given in CapsNet paper
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L,1))

    
