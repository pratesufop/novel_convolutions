import tensorflow as tf
import numpy as np
from utils import load_dataset, prep_pixels, plot_training, umap_plot, plot_featureMaps
from models import get_model
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--conv', help='First Convolutional Layer', choices=['CDC', 'LBPConv', 'ConstrainedCNN' , 'MeDiConv'], required=True)
args = parser.parse_args()


# load dataset
trainX, trainY, testX, testY = load_dataset()
# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)

model = get_model(num_outputs = 10, option = args.conv)
# compile model
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint_filepath = './mnist'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='accuracy',
                mode='max',
                save_best_only=True)

callback_list = [model_checkpoint_callback]

training_output = model.fit(trainX, trainY, validation_split=0.2, batch_size = 128 , 
                                            epochs =  30, steps_per_epoch =  int(len(trainX) / 128 ), 
                                            callbacks= callback_list, shuffle= True)

# plot the training process
plot_training(training_output)

model.load_weights(checkpoint_filepath)

# evaluating the model
loss, acc = model.evaluate(testX,testY)

print('%s : Test Acc. %.2f' % (args.conv, 100*acc))

# low-dimensional plot using UMAP
umap_plot(model, testX, testY)

# showing some feature maps
plot_featureMaps(model, testX[:10])
