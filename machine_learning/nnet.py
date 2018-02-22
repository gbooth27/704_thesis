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
DIM = 6
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
    Run neural net for quantum state predictions.
    :param x: features for training
    :param y: labels for data
    :param gpu:use gpu optimization
    :return: model
    """
    y = np.array([psi.weights for _ in range( (2**DIM))])#[i for i in range(len(psi.weights))]
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
        model.add(Dense(400, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(1000, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(1000, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(2**DIM, kernel_initializer='random_uniform', activation="tanh"))#output_dim = (dim1,dim2)))
        model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Reshape())
        model.add(keras.layers.Reshape((dim1, 1)))

        # Set the optimizer.
        sgd = optimizers.Adam()
        # Compile model.
        model.compile(loss=energy, optimizer=sgd)
        print(model.summary())
    if gpu:
        # Fit the model.
        # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
        model.fit(x, y, epochs=400, batch_size=128, verbose=2 , validation_split=0.2)
    else:
        # Fit the model.
        # Feel free to change this batch size.
        model.fit(x, y, epochs=100, batch_size=4096, verbose =2, validation_split=0.2, callbacks=[CustomMetrics()])
    return model

if __name__ == '__main__':
    model = run_nnet(psi.basis, True, "")
    pred = model.predict(psi.basis)

    min = sp.optimize.minimize(min_energy, psi.weights)
    print("#########################################################")
    res = np.reshape(min.x, (len(min.x), 1))
    norm = np.dot(res.T, res)
    print(min_energy(res))
    #print(res/norm)#np.linalg.norm(np.reshape(min.x, (len(min.x), 1))))
    print("#########################################################")
    #print(str(pred[1]))
    min = min_energy(pred[1])[0][0]
    pos = 0
    # find the best energy of the neural net
    for i in range(len(pred)):
        tmp = min_energy(pred[i])[0][0]
        if tmp < min:
            pos = i
            min = tmp
    print(min_energy(pred[pos]))


