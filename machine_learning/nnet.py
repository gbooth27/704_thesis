from keras.models import Model
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
