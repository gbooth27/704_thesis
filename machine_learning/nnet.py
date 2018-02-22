from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers
import keras
from keras.models import load_model
import machine_learning.generate_basis as wave
import tensorflow as tf
import scipy as sp
from keras import backend as K



import numpy as np
import argparse
from matplotlib import pyplot as plt

import keras.callbacks as cbks
class CustomMetrics(cbks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('energy'):
               print (logs[k])


# https://keras.io/layers/recurrent/
DIM = 10
psi = wave.Psi(DIM, 1)

def energy(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    #sess = tf.InteractiveSession()
    #y= K.sum(y_true, 1)
    ham = K.variable(psi.Hamiltonian.toarray())
    #print(K.shape(ham))
    s = K.shape(y_pred)

    y_pred= K.reshape(y_pred, (s[1], -1))
    #s = K.shape(y_pred)
    un_norm = K.dot(K.dot(y_pred, ham), y_pred)
    #psi.weights[y_true] = y_pred
    #un_norm = np.dot(np.dot(psi.weights.T, psi.Hamiltonian.toarray()), psi.weights)[0][0]
    #norm = un_norm/np.dot(psi.weights.T, psi.weights)[0][0]
    return un_norm

def min_energy(p):
    un_norm = np.dot(np.dot(p.T, psi.Hamiltonian.toarray()), p)
    norm = un_norm / np.dot(p.T, p)
    return norm


def load_net(n):
    """
    Creates a neural network of the correct size for optimization
    :param n: number of particles
    :return:
    """
    return None

def run_nnet(x, gpu, m):
    """
    Run neural net for power predictions.
    :param x: features for training
    :param y: labels for data
    :param gpu:use gpu optimization
    :return: model
    """
    y = psi.weights #[i for i in range(len(psi.weights))]
    if m != "":
        # load the model so that we can continue training
        model = load_model(m)
    else:
        # Create model.
        model = Sequential()
        dim1 = len(x)
        dim2 = len(x[0])
        # Add the layers.
        # Tuning
        model.add(Dense(dim1, input_dim = dim2, kernel_initializer='random_uniform', activation='relu'))
        #model.add(Dense(400, kernel_initializer='random_uniform', activation='relu'))
        #model.add(Dense(200, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(2**DIM, kernel_initializer='random_uniform', activation="relu"))#output_dim = (dim1,dim2)))
        #model.add(keras.layers.Reshape())
        #model.add(keras.layers.Reshape(dim1, 1))


        # Set the optimizer.
        #sgd = optimizers.SGD(lr=0.01, clipnorm=2.)#, momentum=0.1, nesterov=True)
        #sgd = optimizers.Adagrad(clipnorm=2.)
        sgd = optimizers.Adam()
        #sgd = optimizers.Adadelta(clipnorm=2.)
        # Compile model.
        model.compile(loss=energy, optimizer=sgd)
        print(model.summary())
    if gpu:
        # Fit the model.
        # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
        model.fit(x, y, epochs=50, batch_size=512, verbose=2 , validation_split=0.2)
    else:
        # Fit the model.
        # Feel free to change this batch size.
        model.fit(x, y, epochs=100, batch_size=4096, verbose =2, validation_split=0.2, callbacks=[CustomMetrics()])
    return model

if __name__ == '__main__':
    run_nnet(psi.basis, True, "")
    #min = sp.optimize.minimize(min_energy, psi.weights)
    #print(min)
