from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.models import load_model
import machine_learning.generate_basis as wave

import numpy as np
import argparse
from matplotlib import pyplot as plt

# https://keras.io/layers/recurrent/
psi = wave.Psi(6, 1)

def energy(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    un_norm = np.dot(np.dot(psi.weights.T, psi.Hamiltonian.toarray()), psi.weights)[0][0]
    norm = un_norm/np.dot(psi.weights.T, psi.weights)[0][0]
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
    y = psi.weights
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
        model.add(keras.layers.SimpleRNN(dim1, input_dim=dim2, kernel_initializer='random_uniform', activation='relu', return_sequences = True))
        #model.add(Dense(400, kernel_initializer='random_uniform', activation='relu'))
        model.add(keras.layers.SimpleRNN(200, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='random_uniform'))


        # Set the optimizer.
        #sgd = optimizers.SGD(lr=0.01, clipnorm=2.)#, momentum=0.1, nesterov=True)
        #sgd = optimizers.Adagrad(clipnorm=2.)
        sgd = optimizers.Adam()
        #sgd = optimizers.Adadelta(clipnorm=2.)
        # Compile model.
        model.compile(loss=energy, optimizer=sgd)
    if gpu:
        # Fit the model.
        # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
        model.fit(x, y, epochs=50, batch_size=512, verbose=2 , validation_split=0.2)
    else:
        # Fit the model.
        # Feel free to change this batch size.
        model.fit(x, y, epochs=100, batch_size=4096, verbose =2, validation_split=0.2)
    return model

if __name__ == '__main__':
    run_nnet(psi.basis, True, "")
