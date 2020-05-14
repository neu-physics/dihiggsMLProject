import numpy as np
import argparse
import subprocess
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import CapsNet
import os
import sys
#sys.path.insert(0, '/afs/cern.ch/work/l/lian/public/diHiggs/dihiggsMLProject/')
sys.path.insert(0, '/Users/flywire/Desktop/sci/diHiggs/CapsNet')
K.set_image_data_format('channels_last')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float, 
                    help="Learning Rate")
parser.add_argument('--lr_decay', default=0.9, type=float,
                    help="Decay rate of learning rate")
parser.add_argument('--routings', default=3, type=int,
                    help="the number of iterations when running dynamic routing")
parser.add_argument('--debug', action='store_true',
                    help="Save weights by TensorBoard")
parser.add_argument('--decoder', action='store_true',
                    help="whether to use decoder to do image reconstruction")
parser.add_argument('--lam_reco', default=0.392, type=float,
                    help="coefficient of loss of decoder")
parser.add_argument('--save_dir')
parser.add_argument('--test', action='store_true',
                    help="Whether to do testing on the trained model")
parser.add_argument('--train', action='store_true',
                    help="whether to train a model")
parser.add_argument('--weights', default=None,
                    help="The path of saved weights that will be used for testing")


args = parser.parse_args()

if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


#load data
(x_train, y_train), (x_test, y_test) = CapsNet.load_mnist()
decoder = args.decoder

#build model
train_model, eval_model = CapsNet.CapsNet(input_shape=x_train.shape[1:],
                                  n_class=len(np.unique(np.argmax(y_train, 1))), 
                                  n_routings=args.routings, decoder=args.decoder)
train_model.summary()

if(args.train):
    CapsNet.train(model=train_model, data=((x_train,y_train),(x_test,y_test)), args=args)
if(args.test):
    if(args.weights is not None):
        train_model.load_weights(args.weights)
    elif(args.train):
        train_model.load_weights(args.save_dir+'/trained_model.h5')
    else:
        print("No weights are provided to run the test!!!")
        raise KeyboardInterrupt
    CapsNet.test(model=eval_model, data=(x_test,y_test), args=args)
