import rbm.rbm as rbm
import machine_learning.wave_function as wave
import numpy as np
import scipy as sp


def run_rbm(psi):
    """
    Sets up and runs an RBM based on wavefunction PSI
    :param psi: wavefunction
    :return:
    """
    # Create the RBM with N visible and 2N hidden
    net = rbm.RBM(num_visible=psi.size, num_hidden=2*psi.size)
    net.train(psi.basis, max_epochs=1000)
    print(net.weights)
    print('########################################################')
    test = psi.basis[10].reshape((psi.size, 1)).T
    print(test)
    print(net.run_visible(test))
    return net

def construct_wave(net, psi):
    """
    Constructs the wavefunction in the manner outlined in the paper
    "Solving the quantum many-body problem with artificial neural networks"
    by Carleo and Troyer
    :param net:
    :return:
    """
    a= 1
    b = 1
    # iterate over the entire psi
    for n in range(2**psi.size):
        F = 1
        # iterate through each of the M weights
        for i in range(net.weights.shape[1]):
            cosh_mini_sum = 0
            exp_mini_sum = 0
            # iterate over the N spins in each basis
            for j in range(net.weights.shape[0]):
                cosh_mini_sum += net.weights[j][i] * psi.basis[n][j]
                exp_mini_sum += a * psi.basis[n][j]

            cosh_mini_sum+=b
            # do the successive product
            f_i = 2* np.cosh(cosh_mini_sum)
            F *= f_i
        # get the coefficient
        psi_n = np.exp(exp_mini_sum)*F
        psi.weights[n] = psi_n


if __name__ == '__main__':
    psi = wave.Psi(6, 2)
    psi_2 = wave.Psi(6, 2)
    net = run_rbm(psi)
    construct_wave(net,psi)
    print(psi.min_energy(psi.weights))
    #dot = np.dot(psi.weights.T, psi.weights)
    #norm_psi = psi.weights/np.sqrt(dot)
    #print(norm_psi)
    #print(np.dot(norm_psi.T, norm_psi))

    min = sp.optimize.minimize(psi_2.min_energy, psi_2.weights, options={'disp': True})
    print("#########################################################")
    res = np.reshape(min.x, (len(min.x), 1))

    norm = np.dot(res.T, res)
    print("Result: " + str(min.x/norm))
    print(psi_2.min_energy(res))

