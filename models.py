import tensorflow as tf
from novel_layers import CDC, LBPConv, ConstrainedCNN, MeDiConv


def get_model(num_outputs, option):    
    """
    Classifier (Inspired in the Fig3)
    """
    inputs = tf.keras.Input(shape=(28,28,1))
    
    if option == 'LBPConv':
        out = LBPConv(num_filters = 16, kernel_size = (3,3), num_channels = inputs.shape[-1], num_lbc = 32)(inputs)
    elif option == 'MeDiConv':
        out = MeDiConv( num_filters = 16, kernel_size = (3,3), num_channels = inputs.shape[-1])(inputs)
    elif option == 'CDC':
        out = CDC( num_filters = 16, strides = (1,1), kernel = (3,3), theta = 0.7, layer_name = 'CDC', padding = 'SAME')(inputs)
    elif option == 'ConstrainedCNN':
        out = ConstrainedCNN(num_filters = 16, kernel_size = (3,3), num_channels = inputs.shape[-1])(inputs)
    else:
        out = inputs
        
    out = tf.keras.layers.Conv2D( 16, kernel_size = (3,3), padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.activations.relu(out)
    out = tf.keras.layers.MaxPooling2D((2, 2))(out)

    out = tf.keras.layers.Conv2D( 32, kernel_size = (3,3), padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.activations.relu(out)
    out = tf.keras.layers.MaxPooling2D((2, 2))(out)
    
    out = tf.keras.layers.Flatten()(out)
    
    output = tf.keras.layers.Dense(num_outputs, activation = 'softmax',  name = 'cls')(out)
    
    return tf.keras.Model(inputs=inputs, outputs=output, name = 'mnist_cls')