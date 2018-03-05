from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers
import keras
from keras.models import load_model
import machine_learning.wave_function as wave
import tensorflow as tf
import scipy as sp
from keras import backend as K

import machine_learning.generator as gen

import numpy as np
import argparse
from matplotlib import pyplot as plt

import progressbar

DIM = 12
psi = wave.Psi(DIM, 2)

def energy(y_true, y_pred):
    """
    Compute the energy of the system for use as the loss function.
    :param y_true:
    :param y_pred:
    :return:
    """
    # Put hamiltonian into tensor form
    ham = K.variable(psi.Hamiltonian.toarray())
    s = K.shape(y_pred)
    y_pred= K.reshape(y_pred, (s[1], -1))
    # Take the dot product to get energy
    un_norm = K.dot(K.dot(y_pred, ham), y_pred)
    norm = un_norm/K.sum(K.square(y_pred))
    return norm

def energy_tf(y_true, y_pred):
    """
    Compute the energy of the system for use as the loss function.
    :param y_true:
    :param y_pred:
    :return:
    """
    tf_session = K.get_session()
    # Put hamiltonian into tensor form
    ham = K.variable(psi.Hamiltonian.toarray())
    s = K.shape(ham).eval(session=tf_session)
    y_pred= K.reshape(y_pred, (s[1], -1))
    # Take the dot product to get energy
    un_norm = K.dot(K.dot(K.transpose(y_pred), ham), y_pred)
    norm = un_norm/K.sum(K.square(y_pred))
    return norm

def run_nnet(x, gpu, m, backend):
    """
    Run neural net for quantum state predictions.
    :param x: features for training
    :param y: labels for data
    :param gpu:use gpu optimization
    :param backend: whether or not you are using a tensorflow backend
    :return: model
    """
    #y = np.array([psi.weights for _ in range( (2**DIM))])#[i for i in range(len(psi.weights))]
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
        model.add(Dense(DIM, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(2*DIM, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dropout(0.1, noise_shape=None, seed=None))
        model.add(Dense(4*DIM, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dropout(0.1, noise_shape=None, seed=None))
        model.add(Dense(2**DIM, kernel_initializer='random_uniform', activation="relu"))#output_dim = (dim1,dim2)))
        # Normalize the output vector PSI
        model.add(keras.layers.BatchNormalization())
        # Make sure that it is of the correct shape.
        model.add(keras.layers.Reshape((dim1, 1)))

        # Set the optimizer.
        sgd = optimizers.Adam()
        # Compile model.
        if backend:
            model.compile(loss=energy_tf, optimizer=sgd)
        else:
            model.compile(loss=energy, optimizer=sgd)

        print(model.summary())
    if gpu:
        # Fit the model.
        # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
        if backend:
            model.fit_generator(gen.generator_mem(256, DIM), steps_per_epoch=DIM, epochs=10)
            #model.fit(x, y, epochs=10, batch_size=128, verbose=1)
        else:
            model.fit_generator(gen.generator_mem(16, DIM), steps_per_epoch=DIM, epochs=10)
            #model.fit(x, y, epochs=400, batch_size=128, verbose=1 , validation_split=0.2)
    else:
        # Fit the model.
        # Feel free to change this batch size.
        pass
        #model.fit(x, y, epochs=100, batch_size=4096, verbose =1, validation_split=0.2, callbacks=[CustomMetrics()])
    return model