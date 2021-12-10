import tensorflow as tf
import numpy as np

class CDC(tf.keras.layers.Layer):
    def __init__(self, num_filters, strides, kernel, theta, layer_name, padding):
        super(CDC, self).__init__()
        self.num_filters = num_filters
        self.kernel = kernel
        self.theta = theta
        self.strides = strides
        self.layer_name = layer_name
        self.padding = padding
        
    def build(self, input_shape):
        
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        
        self._filter = tf.Variable(initializer(shape=(self.kernel[0],self.kernel[1],  input_shape[-1], self.num_filters)), trainable=True)
        
    def call(self, inputs):
        
        vanilla = tf.nn.conv2d(inputs, self._filter, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding, name=self.layer_name + '/normal')
        
        kernel_diff = tf.reduce_sum(self._filter, axis=0, keepdims=True)
        kernel_diff = tf.reduce_sum(kernel_diff, axis=1, keepdims=True)
        kernel_diff = tf.tile(kernel_diff, [1, 1, 1, 1])
        out_diff = tf.nn.conv2d(inputs, kernel_diff, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding, name=self.layer_name + '/diff')
            
        return  vanilla - self.theta*out_diff


class LBPConv(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, num_channels, num_lbc):
        super(LBPConv, self).__init__()
        
        self.num_lbc = num_lbc
        self.num_filters = num_filters
        self.kernel = kernel_size
        self.num_channels = num_channels
        
    def build(self, input_shape):
        
        initializer = tf.keras.initializers.zeros()

        self.fixed_W = tf.Variable(initializer(shape=(self.kernel[0],self.kernel[1], self.num_channels, self.num_lbc)), trainable = False)

        # randomize every single filter
        rpos = tf.Variable(initializer(shape=(self.kernel[0]*self.kernel[1], self.num_channels, self.num_lbc), dtype=tf.int32))
        for i in np.arange(self.num_lbc):
            for j in np.arange(self.num_channels):
                pos = tf.random.shuffle(tf.cast(tf.range(0,self.kernel[0]*self.kernel[1],1), dtype=tf.int32))
                for enum,p in enumerate(pos):
                    rpos[enum, j, i].assign(p)
        rpos = tf.reshape(rpos, [self.kernel[0],self.kernel[1],self.num_channels,self.num_lbc])
        
        self.fixed_W.assign(tf.where(rpos <=1, -tf.ones(rpos.shape), tf.zeros(rpos.shape)) + tf.where(rpos >=7, tf.ones(rpos.shape), tf.zeros(rpos.shape)))
        
        self.learned_W = tf.Variable(tf.keras.initializers.glorot_normal()(shape=(1,1, self.num_lbc, self.num_filters)))
        
    def call(self, inputs):
        
        output = tf.nn.conv2d(inputs, self.fixed_W , strides=[1,1,1,1], padding='SAME')
        
        #1x1 convolutions
        output = tf.keras.activations.relu(output)

        output = tf.nn.conv2d(output, self.learned_W  , strides=[1,1,1,1], padding='SAME')
        
        return  output

    
class ConstrainedCNN(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, num_channels):
        super(ConstrainedCNN, self).__init__()
        
        self.num_filters = num_filters
        self.kernel = kernel_size
        self.num_channels = num_channels

    def build(self, input_shape):
        
        initializer = tf.keras.initializers.glorot_normal()
        self.W = tf.Variable(initializer(shape=(self.kernel[0],self.kernel[1], self.num_channels, self.num_filters)))
        
    def call(self, inputs):
        
        # See Algorithm 1 from the paper
        # Do feedforward pass
        output = tf.nn.conv2d(inputs, self.W , strides=[1,1,1,1], padding='SAME')

        #Set the filters (center) to zero
        for i in np.arange(self.W.shape[-1]):
            for j in np.arange(self.num_channels):
                self.W[int(np.floor(self.kernel[0]/2)), int(np.floor(self.kernel[1]/2)),j,i].assign(0) 
        
        # Normalize (L1 Norm)
        norms = tf.reduce_sum(self.W, [0,1,2], keepdims=True)
        
        self.W.assign(self.W/(norms + tf.keras.backend.epsilon()))
        
        #Set the central values to (-1)
        for i in np.arange(self.W.shape[-1]):
            for j in np.arange(self.num_channels):
                self.W[int(np.floor(self.kernel[0]/2)), int(np.floor(self.kernel[1]/2)),j,i].assign(-1) 
        
        
        return  output

    
class MeDiConv(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, num_channels):
        super(MeDiConv, self).__init__()
        
        self.num_filters = num_filters
        self.kernel = kernel_size
        self.num_channels = num_channels

    def build(self, input_shape):
        
        self.conv = tf.keras.layers.Conv2D(self.num_filters, self.kernel, use_bias= False, padding='same')
        
    def call(self, inputs):

        patches = tf.image.extract_patches(images=inputs,
                                   sizes=[1, self.kernel[0],  self.kernel[1], 1],
                                   strides=[1, 1, 1, 1],
                                   rates=[1, 1, 1, 1],
                                   padding='SAME')

        output = tf.nn.top_k(patches, k =  int(np.floor(patches.shape[-1]/2)) +1, sorted = True).values[:,:,:,-1][:, :,:,tf.newaxis]
        
        mean_ = tf.reduce_mean(output, axis=(0,1,2), keepdims= True)

        output -= mean_
        
        output = self.conv(output)
        
        return  output