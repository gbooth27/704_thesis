import rbm.rbm as rbm
import machine_learning.wave_function as wave
import numpy as np
import scipy as sp
import progressbar

N = 3
M = N*N
psi = wave.Psi(N, 2)
psi_2 = wave.Psi(N, 2)


def run_rbm(psi_):
    """
    Sets up and runs an RBM based on wavefunction PSI
    :param psi_: wavefunction
    :return:
    """
    # Create the RBM with N visible and 2N hidden
    net = rbm.RBM(num_visible=psi_.size, num_hidden=2 * psi_.size)
    net.train(psi_.basis, max_epochs=1000)
    print(net.weights)
    print('########################################################')
    test = psi_.basis[10].reshape((psi_.size, 1)).T
    print(test)
    print(net.run_visible(test))
    return net


def energy_function(params):
    """

    :param params:
    :return:
    """
    val = N * M
    params_copy = params.reshape(((N * M) + N + M, 1))
    wieghts = np.array(params_copy[0:N*M])

    wieghts = np.reshape(wieghts, (N,M))



    #wieghts = np.reshape(np.array(w[0:val]),(N,M))
    #params_copy = np.copy(params).reshape(((N * M) + N + M, 1))

    A = params_copy[N*M:(N*M)+N]
    B = params_copy[(N*M)+N:]
    # get wave then energy
    wave = construct_wave(wieghts, psi, A, B)
    un_norm = np.dot(np.dot(wave.T, psi.Hamiltonian.toarray()), wave)
    #norm = un_norm / np.dot(wave.T, wave)
    return un_norm



def construct_wave(weights, psi_target, a, b):
    """
    Constructs the wavefunction in the manner outlined in the paper
    "Solving the quantum many-body problem with artificial neural networks"
    by Carleo and Troyer
    :param net:
    :return:
    """
    # iterate over the entire psi
    #bar = progressbar.ProgressBar()
    for n in range(2 ** psi_target.size):
        F = 1
        # iterate through each of the M weights
        for i in range(weights.shape[1]):
            cosh_mini_sum = 0
            exp_mini_sum = 0
            # iterate over the N spins in each basis
            for j in range(weights.shape[0]):
                cosh_mini_sum += weights[j][i] * psi_target.basis[n][j]
                exp_mini_sum += a[j] * psi_target.basis[n][j]

            cosh_mini_sum+=b[i]
            # do the successive product
            f_i = 2* np.cosh(cosh_mini_sum)
            F *= f_i
        # get the coefficient
        psi_n = np.exp(exp_mini_sum)*F
        psi_target.weights[n] = psi_n
    return psi_target.weights/np.sqrt(np.dot(psi_target.weights.T, psi_target.weights))


if __name__ == '__main__':

    #net = run_rbm(psi)
    #construct_wave(net, psi)
    #print(psi.min_energy(psi.weights))
    #dot = np.dot(psi.weights.T, psi.weights)
    #norm_psi = psi.weights/np.sqrt(dot)
    #print(norm_psi)
    #print(np.dot(norm_psi.T, norm_psi))
    params = np.ones(((N*M)+N+M, 1), dtype=np.float128)
    print(energy_function(params))
    min_rbm = sp.optimize.minimize(energy_function, params, options={'disp': True})
    print("Result: " + str(min_rbm.x))
    print(energy_function(min_rbm.x))

    print("#########################################################")


    min = sp.optimize.minimize(psi_2.min_energy, psi_2.weights, options={'disp': True})
    print("#########################################################")
    res = np.reshape(min.x, (len(min.x), 1))

    norm = np.dot(res.T, res)
    print("Result: " + str(min.x/norm))
    print(psi_2.min_energy(res))
