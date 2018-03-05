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

DIM = 9
psi = wave.Psi(DIM, 2)
psi.get_ground()

def min_energy(p):
    un_norm = np.dot(np.dot(p.T, psi.Hamiltonian.toarray()), p)
    norm = un_norm / np.dot(p.T, p)
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
        model.add(Dropout(0.1, noise_shape=None, seed=None))
        model.add(Dense(DIM//2, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='random_uniform', activation="relu"))#output_dim = (dim1,dim2)))


        # Set the optimizer.
        sgd = optimizers.Adam()
        # Compile model.
        if backend:
            model.compile(loss="mse", optimizer=sgd)
        else:
            model.compile(loss="mse", optimizer=sgd)

        print(model.summary())
    if gpu:
        # Fit the model.
        # DO NOT CHANGE GPU BATCH SIZE, CAN CAUSE MEMORY ISSUES
        if backend:
            model.fit_generator(gen.generator_precompute(128, psi), steps_per_epoch=DIM, epochs=100)
            #model.fit(x, y, epochs=10, batch_size=128, verbose=1)
        else:
            model.fit_generator(gen.generator_precompute(128, psi), steps_per_epoch=DIM, epochs=10)
            #model.fit(x, y, epochs=400, batch_size=128, verbose=1 , validation_split=0.2)
    else:
        # Fit the model.
        # Feel free to change this batch size.
        pass
        #model.fit(x, y, epochs=100, batch_size=4096, verbose =1, validation_split=0.2, callbacks=[CustomMetrics()])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', "-m", dest='model', action='store', required=True, help="path to model being used")
    parser.add_argument('--tensorflow', "-tf", dest='tf', action='store_true', help="if using tensorflow backend")
    args = parser.parse_args()


    model = run_nnet(psi.basis, True, "", args.tf)
    pred = model.predict(psi.basis)


    #print(res/norm)#np.linalg.norm(np.reshape(min.x, (len(min.x), 1))))
    print("#########################################################")
    #print(str(pred[1]))
    min = min_energy(pred)[0][0]
    pos = 0
    # find the best energy of the neural net
    print(min)
    actual_min = min_energy(psi.ground)
    print(actual_min)

    print(min/actual_min)

