import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import umap as umap
import umap.plot
import numpy as np


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def plot_training(training_output):
    plt.figure(figsize=(20,10))

    plt.subplot(121)
    plt.plot(training_output.history['accuracy'])
    plt.plot(training_output.history['val_accuracy'])
    plt.title('Acc. Epochs')
    plt.legend(['Acc','Val. Acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Acc')

    plt.subplot(122)
    plt.plot(training_output.history['loss'])
    plt.plot(training_output.history['val_loss'])
    plt.title('Loss Epochs')
    plt.legend(['Loss','Val. Loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig('training.png')

def umap_plot(model, testX, testY):
    feat_model = tf.keras.Model(inputs = model.inputs, outputs= model.get_layer(index = -2).output)
    all_feats = feat_model.predict(testX)

    classes = np.unique(np.argmax(testY, axis=1))
    embedding = umap.UMAP(n_neighbors=11).fit_transform(np.array(all_feats))

    _, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*embedding.T, s=20.0, c=np.argmax(testY, axis=1), cmap='jet_r', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title('UMAP Embedding', fontsize=14)
    cbar = plt.colorbar(boundaries=np.arange(len(classes)+1)-0.5)
    cbar.set_ticks(np.arange(len(classes)))
    cbar.set_ticklabels(classes)
    plt.tight_layout()
    
    plt.savefig('umap_plot.png')
    
    
def plot_featureMaps(model, data):
    
    feat_model = tf.keras.Model(inputs = model.inputs, outputs= model.get_layer(index=1).output)
    featmaps = feat_model.predict(data)
    plt.figure(figsize=(20,20))

    num_filters = model.get_layer(index=1).output.shape[-1]

    idx= np.random.choice(np.arange(len(featmaps)))

    for i in np.arange(num_filters):
        plt.subplot(4,4,i+1)
        plt.imshow(featmaps[idx,:,:,i])
    
    plt.savefig('feature_maps.png')
    
