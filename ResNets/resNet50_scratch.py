import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Add, Dense, ZeroPadding2D, Activation, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import os, random

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



def resnet_50(input_shape = (244, 244, 3), classes = 2):

    '''
    input_shape - dimensions of the image
    '''

    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input)

    # Initial Convolution Block
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((1, 1), strides=(1, 1))(X)

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

    X = AveragePooling2D((1,1))(X)

    # Flatten and create model
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs= X_input, outputs = X, name= "ResNet50")

    return model

def plot_training_history(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['loss'])
    plt.title('Model accuracy and loss')
    plt.ylabel('Accuracy/Loss')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.show()


def main():
    data = np.genfromtxt("SimpleDataSet.csv", delimiter=",")
    y = data[:,-1]
    X = data[:,:3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)

    inner = 16

    # one hot encoding labels
    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)

    # change input dimensions so it has 4 dimensions instead of 2
    X_train = np.reshape(X_train, (X_train.shape[0],1,1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0],1,1,X_test.shape[1]))
    train_zeros = np.zeros((X_train.shape[0],inner,inner,X_train.shape[1]))
    test_zeros = np.zeros((X_test.shape[0],inner,inner,X_test.shape[1]))
    X_train = X_train + train_zeros
    X_test = X_test + test_zeros

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape)) # (size, 64, 64, 3) (1080, 64, 64, 3)
    print("Y_train shape: " + str(y_train_onehot.shape)) # (size, number of labels) (1080, 6)
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(y_test_onehot.shape))

    model = resnet_50(input_shape=(inner,inner,3), classes=2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    history = model.fit(X_train, y_train_onehot, epochs=3, batch_size=32)
    preds = model.evaluate(X_test, y_test_onehot)
    print(preds)
    print("Loss = ", preds[0])
    print("Test Accuracy = ", preds[1])

    plot_training_history(history)


    # true_model = resnet50.ResNet50(include_top=True, weights='imagenet')
    # true_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    # true_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # preds = true_model.predict(X_test)
    # print(preds)
    # print("\n-----------\n")
    # print(y_test)



if __name__ == '__main__':
    main()


# # Ways to test the identity block and conv block code
#
# tf.reset_default_graph()
#
# with tf.Session() as test:
#     np.random.seed(1)
#     A_prev = tf.placeholder("float", [3, 4, 4, 6])
#     X = np.random.randn(3, 4, 4, 6)
#     A = conv_block(A_prev, filters=[2, 4, 6], kernel_size=2)
#     A = identity_block(A_prev, filters=[2, 4, 6], kernel_size=2)
#     test.run(tf.global_variables_initializer())
#     out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#     print("out = " + str(out[0][1][1][0]))


