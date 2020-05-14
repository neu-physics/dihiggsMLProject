# Reference : https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')



def squash(v, axis=-1):
    """
    Squash function: The non-linear activation function used in Capsule layers
    """
    s_norm_squared = K.sum(K.square(v), axis, keepdims=True)
    scale = s_norm_squared/(1 + s_norm_squared)/K.sqrt(s_norm_squared + K.epsilon())
    return scale * v

class Squash(layers.Layer):
    def call(self, inputs, **kwargs):
        return squash(inputs)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = super(Squash, self).get_config()
        return config

class Length(layers.Layer):
    """
    Compute the length of vectors.
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
    
    def compute_output_shape(self, input_shape):
        #The last dimension if summed over so omit the last dimension
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config
    
class Mask(layers.Layer):
    """
    This layer is used to expand all vectors in the capsule to scalars
    If the y is given, then use given y to mask the digit layer and expand that to FC
    """
    def call(self, inputs, **kwargs):
        if (type(inputs) is list):    #the true label is provided shape=[None, n_classses]
            assert len(inputs)==2
            inputs, mask = inputs
        else: # no true label provided, mask the digit layer by the prediction with largest length
            # input.shape=[None, num_cpasule, dim_capsule]
            # mask.shape = [None, num_capsule]
            # calculate the length of each capsule in last layer
            # length.shape = [None, num_capsule]
            length = K.sqrt(K.sum(K.square(inputs), -1))
            
            mask = K.one_hot(indices=K.argmax(length,1), num_classes=length.get_shape().as_list()[1])
        
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked
    
    def compute_output_shape(self, input_shape):
        # output_shape = [None, num_capsule*dim_capsule]
        if (type(input_shape[0]) is tuple): # true label provided
            # input_shape[0] is capsules and input_shape[1] is true lables
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else: # true label not provided
            return tuple([None, input_shape[1] * input_shape[2]])
        
    def get_config(self):
        config = super(Mask, self).get_config()
        return config
    


class CapsuleLayer(layers.Layer):
    def __init__(self, n_capsule, dim_capsule, n_routings, kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.n_capsule = n_capsule
        self.dim_capsule = dim_capsule
        self.n_routings = n_routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        assert len(input_shape) >= 3, "Input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(name='W',
                                 shape=[self.n_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer
                                )
        super(CapsuleLayer, self).build(input_shape)
        #self.built = True
        
    def call(self, inputs):
        # previous inputs.shape=[None, num_capsule_input, dim_capsule_input]
        # after expansion: inputs.shape=[None, 1, num_capsule_input, dim_capsule_input]
        inputs_expand = tf.expand_dims(inputs,1)
        
        # Copy inputs `num_capsule` times so it will multiply with different W several times to generate num_capsule capsules in the next layer
        # inputs_tiled.shape = [None, num_capsule, num_capsule_input, dim_capsule_input]
        inputs_tiled = tf.tile(inputs_expand, [1, self.n_capsule, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)
        
        # Compute inputs*W 
        # x.shape = [num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape = [num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as batch, then multiply [input_dim_capsule]*[dim_capsule, input_dim_capsule]
        # Do the multiplication several times since inputs are tiled so will result in several capsules in next layer
        u_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)
        
        
        # routing:
        # Initialize b as 0 and apply softmax on it.
        # Then do agreement a = v*u_hat
        # update b = b + a
        # b.shape = [None, num_capsule, input_num_capsule, 1, 1]
        b = tf.zeros(shape=[K.shape(u_hat)[0], self.n_capsule, self.input_num_capsule, 1, 1])
        
        assert self.n_routings > 0, 'The routings should be > 0'
        for i in range(self.n_routings):
            # do softmax for b with num_capsule axis
            c = tf.nn.softmax(b, axis=1)
            # c.shape = [None, num_capsule, input_num_capsule, 1, 1]
            # u_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
            outputs = tf.multiply(c, u_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)
            # outputs.shape = [None, num_capsule, dim_capsule]
            
            if i < self.n_routings-1:
                # outputs.shape = [None, num_capsule, dim_capsule]
                # u_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
                #b += K.batch_dot(outputs_n, u_hat, [2,3])
                outputs_tiled = tf.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
                agreement = tf.matmul(u_hat, outputs_tiled, transpose_a=True)
                b = tf.add(b, agreement)
                
        outputs = tf.squeeze(outputs, [2,4])
        return outputs
    
    def compute_output_shape(self, input_shape):
        return tuple([None, self.n_capsule, self.dim_capsule])
    
    def get_config(self):
        config = {
            'n_capsule': self.n_capsule,
            'dim_capsule': self.dim_capsule,
            'n_routings': self.n_routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
        
def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    '''
    output shape: [None, num_capsule, dim_capsule]
    '''
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding = padding, name='conv1_caps')(inputs)
    n_capsules = output.shape[1]*output.shape[2]*output.shape[3]//dim_capsule
    outputs = layers.Reshape(target_shape=[int(n_capsules),dim_capsule], name="primarycap_reshape")(output)
    outputs = Squash(name="primary_squash")(outputs)
    return outputs