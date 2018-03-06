import numpy as np
import random


def generator_mem(batch_size, n):
    while (True):
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            # generate the labels
            psi = [1 for _ in range(2 ** n)]
            batch_y.append(np.asarray(psi).reshape((len((psi)), 1)))
            # generate a random sample of basis vectors
            state = [np.random.choice([0,1]) for _ in range(n)]
            #s =np.asarray(state)#.reshape(n,1)
            #p = s.reshape((n,1))
            #sha = p.shape
            batch_x.append(np.asarray(state))#.reshape((n,1)))
        yield np.array(batch_x), np.array(batch_y)


def generator_precompute(batch_size, wave):
    while (True):
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            # generate the labels


            # generate a random sample of basis vectors
            #state = [np.random.choice([0,1]) for _ in range(wave.size)]
            state = random.choice(wave.basis)
            num = wave.binary_to_int(state)

            #s =np.asarray(state)#.reshape(n,1)
            #p = s.reshape((n,1))
            #sha = p.shape
            batch_x.append(np.asarray(state))#.reshape((n,1)))
            batch_y.append(wave.ground[num])
        yield np.array(batch_x), np.array(batch_y)


def generator_approx(batch_size, wave):
    while (True):
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            # generate the labels


            # generate a random sample of basis vectors
            #state = [np.random.choice([0,1]) for _ in range(wave.size)]
            state = random.choice(wave.basis)
            num = wave.binary_to_int(state)

            #s =np.asarray(state)#.reshape(n,1)
            #p = s.reshape((n,1))
            #sha = p.shape
            batch_x.append(np.asarray(state))#.reshape((n,1)))
            batch_y.append(wave.collapsed[num])
        yield np.array(batch_x), np.array(batch_y)