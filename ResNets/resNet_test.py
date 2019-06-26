import numpy as np
import tensorflow as tf
from keras.layers import Input, Add, Dense, ZeroPadding2D, Activation, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# TODO: kernal initalizer?
# TODO: average pooling instead of max pooling for final layer?

def identity_block(X, filters, kernel_size):
    '''
    residual block with 3 skips

    X - input tensor
    filters - number of filters in the convolutional layer
    kernel_size - dimension of square filter to go over image
    stage - way to label position of block in network
    '''

    f1, f2, f3 = filters
    X_shortcut = X

    # First block
    X = Conv2D(filters=f1, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second Block
    X = Conv2D(filters=f2, kernel_size=(kernel_size,kernel_size), strides=(1,1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third Block
    X = Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    # Add shortcut Block
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X

def conv_block(X, filters, kernel_size, strides=2):
    '''

    residual block with 3 skips, the "shortcut" path has a conv layer

    X - input tensor of shape (h_previous, w_previous, c_previous)
    filters - number of filters in the conv layer
    kernel_size - dimension of square filter to go over image
    strides - how big of a translation the filters taken when going through image

    returns: tensor of shape (h,w,c)
    '''

    f1, f2, f3 = filters

    X_shortcut = X

    # First Block
    X = Conv2D(filters= f1, kernel_size=(1,1), strides=(strides,strides), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second Block
    X = Conv2D(filters=f2, kernel_size=(kernel_size,kernel_size), strides=(1,1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third Block
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)


    # Conv Shortcut and Adding
    X_shortcut = Conv2D(filters= f3, kernel_size=(1,1), strides=(strides,strides), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X



def resnet_50(input_shape = (244, 244, 3)):

    '''
    input_shape - dimensions of the
    '''

    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input)

    # Initial Convolution Block
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # First Block
    X = conv_block(X, filters=[64, 64, 256], kernel_size=3, strides=1)
    X = identity_block(X, filters=[64, 64, 256], kernel_size=3)
    X = identity_block(X, filters=[64, 64, 256], kernel_size=3)


    # Second Block
    X = conv_block(X, filters=[128, 128, 512], kernel_size=3)
    X = identity_block(X, filters=[128, 128, 512], kernel_size=3)
    X = identity_block(X, filters=[128, 128, 512], kernel_size=3)
    X = identity_block(X, filters=[128, 128, 512], kernel_size=3)


    # Third Block
    X = conv_block(X, filters=[256, 256, 1024], kernel_size=3)
    X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
    X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
    X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
    X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)
    X = identity_block(X, filters=[256, 256, 1024], kernel_size=3)

    # Fourth Block
    X = conv_block(X, filters=[512, 512, 2048], kernel_size=3)
    X = identity_block(X, filters=[512, 512, 2048], kernel_size=3)
    X = identity_block(X, filters=[512, 512, 2048], kernel_size=3)

    X = AveragePooling2D((2,2))(X)

    # Flatten and create model
    X = Flatten()(X)
    X = Dense(activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs= X_input, outputs = X, name= "ResNet50")

    return model


# Ways to test the identity block and conv block code

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = conv_block(A_prev, filters=[2, 4, 6], kernel_size=2)
    A = identity_block(A_prev, filters=[2, 4, 6], kernel_size=2)
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))


