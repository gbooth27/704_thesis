from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.models import load_model

import numpy as np
import argparse
from matplotlib import pyplot as plt

# https://keras.io/layers/recurrent/

def load_net(n):
    """
    Creates a neural network of the correct size for optimization
    :param n: number of particles
    :return:
    """
    return None

def run_nnet(x, y, gpu, m):
    """
    Run neural net for power predictions.
    :param x: features for training
    :param y: labels for data
    :param gpu:use gpu optimization
    :return: model
    """
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
        model.add(Dense(dim1, input_dim=dim2, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(400, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(200, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='random_uniform'))


        # Set the optimizer.
        #sgd = optimizers.SGD(lr=0.01, clipnorm=2.)#, momentum=0.1, nesterov=True)
        #sgd = optimizers.Adagrad(clipnorm=2.)
        sgd = optimizers.Adam()
        #sgd = optimizers.Adadelta(clipnorm=2.)
        # Compile model.
        model.compile(loss='mse', optimizer=sgd, metrics=["mae", percent_err])
    if gpu:
        # Fit the model.
        # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
        model.fit(x, y, epochs=50, batch_size=512, verbose=2 , validation_split=0.2)
    else:
        # Fit the model.
        # Feel free to change this batch size.
        model.fit(x, y, epochs=100, batch_size=4096, verbose =2, validation_split=0.2)
    return model

