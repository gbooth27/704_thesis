import rbm.rbm as rbm
import machine_learning.wave_function as wave
import numpy as np
import scipy as sp
import progressbar
from matplotlib import pyplot as plt

N = 4
M = N
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

def build_jac(X):
    """
    builds the jacobian
    :param x:
    :return:
    """
    # each row corresponds to one coefficient
    jac = np.zeros((2**N,((N * M) + N + M )))
    params_copy = X.reshape(((N * M) + N + M, 1))
    X = X.reshape(((N * M) + N + M, 1))
    wieghts = np.array(params_copy[0:N * M])

    wieghts = np.reshape(wieghts, (N, M))

    # wieghts = np.reshape(np.array(w[0:val]),(N,M))
    # params_copy = np.copy(params).reshape(((N * M) + N + M, 1))

    A = params_copy[N * M:(N * M) + N]
    B = params_copy[(N * M) + N:]
    # get wave then energy
    wave = construct_wave(wieghts, psi, A, B)
    wave = (wave)
    for x in range (2**N):
        string = "{0:0"+str(N)+"b}"
        spin = string.format(x)

        for y in range((N * M) + N + M):

            # derivative wrt weights
            if y < (N*M):
                sigma = int(spin[y%N])
                tan_sum = 0
                for i in range(N):
                    tan_sum += X[i] * int(spin[i])


                jac[x][y] = sigma*np.tanh(X[y] + tan_sum)  *wave[x]

            # a
            elif y <(N * M) + N :
                sigma = int(spin[y-(N * M)])
                jac[x][y] = sigma *wave[x]


            # b
            else:
                tan_sum = 0
                for i in range(N):
                    tan_sum += X[i] * int(spin[i])

                jac[x][y] = np.tanh(X[y] + tan_sum) * wave[x]


    jac = (jac)


    ham = psi.Hamiltonian
    jac_f = (np.dot(jac.T, psi.Hamiltonian.dot(wave)))
    jac_int = (np.dot(wave.T, psi.Hamiltonian.dot(jac)).T)
    d_hi =  (np.add(jac_f, jac_int))
    lo= (np.dot(wave.T,wave))
    #if lo[0] == 0:
        #lo =1
    d_lo = (np.dot(jac.T, wave) + np.dot(wave.T, jac).T)
    hi = (np.dot(wave.T, psi.Hamiltonian.dot(wave)))

    end = (np.ndarray.flatten(np.add(lo*d_hi, -hi*d_lo)/(lo**2)))
    #for i in range(len(end)):
        #if end[i] == 0:
            #end[i] = 1
    #print(end)
    return end


def energy_function(params):
    """
    get the energy of the RBM system by constructing the wave then finding energy
    with the hamiltonian
    :param params:
    :return:
    """


    params_copy = params.reshape(((N * M) + N + M, 1))
    #except ValueError:
        #params_copy = params[1].reshape(((N * M) + N + M, 1))
    wieghts = np.array(params_copy[0:N*M])

    wieghts = np.reshape(wieghts, (N,M))

    #wieghts = np.reshape(np.array(w[0:val]),(N,M))
    #params_copy = np.copy(params).reshape(((N * M) + N + M, 1))

    A = params_copy[N*M:(N*M)+N]
    B = params_copy[(N*M)+N:]
    # get wave then energy
    wave = construct_wave(wieghts, psi, A, B)
    wave = (wave)
    un_norm = (np.dot(wave.T, psi.Hamiltonian.dot(wave)))
    norm = un_norm /(np.dot(wave.T, wave))
    return (norm)



def construct_wave(weights, psi_target, a, b):
    """
    Constructs the wavefunction in the manner outlined in the paper
    "Solving the quantum many-body problem with artificial neural networks"
    by Carleo and Troyer
    :param weights: weights of the RBM to be optimized
    :param psi_target: target wavefunction to optimize against
    :param a: visible layers
    :param b: hidden layers
    :return:
    """
    # iterate over the entire psi
    #bar = progressbar.ProgressBar()
    for n in range(2 ** psi_target.size):
        F = 1
        cosh_mini_sum = 0
        exp_mini_sum = 0
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
    return psi_target.weights


if __name__ == '__main__':

    #net = run_rbm(psi)
    #construct_wave(net, psi)
    #print(psi.min_energy(psi.weights))
    #dot = np.dot(psi.weights.T, psi.weights)
    #norm_psi = psi.weights/np.sqrt(dot)
    #print(norm_psi)
    #print(np.dot(norm_psi.T, norm_psi))
    actual = psi_2.diag()
    x = []
    y = []
    #info = np.core.getlimits._float128_ma
    y_1 = [actual for _ in range(2*N)]
    #bar = progressbar.ProgressBar()
    for i in range(1, 2*N):
        M = i
        params = np.random.rand(((N*M)+N+M))/10000#,), dtype=np.float128)
        check = sp.optimize.check_grad(energy_function, build_jac, params)
        print ("Grad Check: "+str(check))
        #print(energy_function(params))
        min_rbm = sp.optimize.minimize(energy_function, params, jac = build_jac, method='CG',
                                        options={'disp': True})
        #print("Result: " + str(min_rbm.x))
        y_i = energy_function(min_rbm.x)[0][0]
        print("Parameters: "+str(params))
        print("Gradient: "+ str(build_jac(params)))
        y.append(y_i)
        x.append(i)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Number of Hidden Units')
    ax.set_ylabel('Percent Error from True Energy')
    #for i in range(len(y)):
    #    y[i] = np.log(y[i])
    #l = np.log(np.asarray(y))
    y_new = [((y[i]-y_1[i])/y_1[i])*100 for i in range(len(y))]
    ax.plot(x, np.abs(y_new), color='b', marker='o', linestyle='solid',
        linewidth=2, markersize=5)
    #ax.plot(x, np.log(np.abs(y_1)), "g")
    xmarks = [i for i in range(1, 2*N, 1)]
    plt.xticks(xmarks)
    plt.show()

    print("#########################################################")


    #min = sp.optimize.minimize(psi_2.min_energy, psi_2.weights, method='BFGS',options={'disp': True})
    #print("#########################################################")
    #res = np.reshape(min.x, (len(min.x), 1))

    #norm = np.dot(res.T, res)
    #print("Result: " + str(min.x/norm))
   # print(psi_2.min_energy(res))
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print(psi_2.diag())

