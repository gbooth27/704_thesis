
import thesis.machine_learning.wave_function as wave
import numpy as np
import scipy as sp
import progressbar
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import thesis.hamiltonian.tfim as tfim

N = 7
M = N
H= 1
psi = None#wave.Psi(N, H)
psi_2 = None #wave.Psi(N, H)


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
                if sigma == 0:
                    sigma = -1
                tan_sum = 0
                for i in range(N):
                    tan_sum += X[i] * int(spin[i])


                jac[x][y] = sigma*np.nan_to_num(np.tanh(X[y] + tan_sum)  *wave[x])

            # a
            elif y <(N * M) + N :
                sigma = int(spin[y-(N * M)])
                if sigma == 0:
                    sigma = -1
                jac[x][y] = sigma *wave[x]


            # b
            else:
                tan_sum = 0
                for i in range(N):
                    tan_sum += X[i] * int(spin[i])

                jac[x][y] = np.nan_to_num(np.tanh(X[y] + tan_sum) * wave[x])


    jac = (jac)


    ham = psi.Hamiltonian
    jac_f = (np.dot(jac.T, psi.Hamiltonian.dot(wave)))
    jac_int = (np.dot(wave.T, psi.Hamiltonian.dot(jac)).T)
    d_hi =  np.nan_to_num(np.add(jac_f, jac_int))
    lo= np.nan_to_num(np.dot(wave.T,wave))
    #if lo[0] == 0:
        #lo =1
    d_lo = np.nan_to_num(np.dot(jac.T, wave) + np.dot(wave.T, jac).T)
    hi = np.nan_to_num(np.dot(wave.T, psi.Hamiltonian.dot(wave)))

    end = (np.ndarray.flatten(np.add(lo*d_hi, -hi*d_lo)/(lo**2)))
    #for i in range(len(end)):
        #if end[i] == 0:
            #end[i] = 1
    #print(end)
    return np.nan_to_num(end)


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
        #cosh_mini_sum = 0
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
            f_i = np.nan_to_num(2* np.cosh(cosh_mini_sum))
            F =np.nan_to_num(F* f_i)
        # get the coefficient
        psi_n = np.nan_to_num(np.exp(exp_mini_sum)*F)
        psi_target.weights[n] = psi_n
    return psi_target.weights

def tfim_builder(N):
    out_filename_base = "thesis/hamiltonian/matrix"
    L = [N]
    PBC = True
    J = 1.0
    # Set up file formatting
    ##################################
    width = 25
    precision = 16
    header = tfim.build_header(L, PBC, J)
    ##################################

    # Build lattice and basis
    ###################################
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)
    ###################################

    # Compute diagonal matrix elements
    ###################################
    print('\tBuilding diagonal matrices...')
    Mz_ME, Ms_ME = tfim.z_magnetizations_ME(lattice, basis)
    JZZ_ME, ZZ_ME = tfim.z_correlations_NN_ME(lattice, basis, J)

    # Write to disk
    columns = ['JZZ', 'ZZ', 'Mz', 'Ms']
    diagonal_arr = np.array([JZZ_ME, ZZ_ME, Mz_ME, Ms_ME]).T
    diag_filename = out_filename_base + tfim.diag_ME_suffix
    col_labels = ''.join(['{:>{width}}'.format(tfim.phys_labels[key],
                                               width=(width + 1)) for key in columns])[3:]
    print("\tWriting diagonal matrix elements to {}".format(diag_filename))
    np.savetxt(diag_filename, diagonal_arr, header=(header + col_labels),
               fmt='%{}.{}e'.format(width, precision - 1))
    ###################################

    # Compute off-diagonal matrix elements
    ###################################
    print('\tBuilding off-diagonal matrices...')
    Mx = tfim.build_Mx(lattice, basis)

    # Write to disk
    Mx_filename = out_filename_base + tfim.Mx_suffix
    print("\tWriting off-diagonal matrix to {}".format(Mx_filename))
    tfim.save_sparse_matrix(Mx_filename, Mx)
    ###################################


if __name__ == '__main__':

    n_s = [4]
    h_s = [2]
    bar_1 = progressbar.ProgressBar()
    bar_2 = progressbar.ProgressBar()
    for h in bar_1(h_s):
        for n in bar_2(n_s):
            N = n
            H = h
            tfim_builder(N)
            psi = wave.Psi(N, H)
            psi_2 = wave.Psi(N, H)

            actual = psi_2.diag()
            x = []
            y = []
            #info = np.core.getlimits._float128_ma
            y_1 = [actual for _ in range(2*N)]
            #bar = progressbar.ProgressBar()
            for i in range(1, N):
                M = i
                params = np.random.rand(((N*M)+N+M))/1000#,), dtype=np.float128)
                #check = sp.optimize.check_grad(energy_function, build_jac, params)
                #print ("Grad Check: "+str(check))
                #jac = build_jac,
                #print(energy_function(params))
                min_rbm = sp.optimize.minimize(energy_function, params,  method='BFGS', options={'disp': True, 'maxiter': 10000000})#,
                                               #jac=build_jac)
                                                #options={'disp': True, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08,
                                                         #'return_all': False, 'maxiter': None})
                #print("Result: " + str(min_rbm.x))
                y_i = energy_function(min_rbm.x)[0][0]
                y.append(y_i)
                x.append(i)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_yscale('log')
            ax.set_xlabel('Number of Hidden Units')
            ax.set_ylabel('Percent Error from True Energy')
            y_new = [((y[i]-y_1[i])/y_1[i])*100 for i in range(len(y))]
            ax.plot(x, np.abs(y_new), color='b', marker='o', linestyle='solid',
                linewidth=2, markersize=5)
            #ax.plot(x, np.log(np.abs(y_1)), "g")
            xmarks = [i for i in range(1, N, 1)]
            plt.xticks(xmarks)
            #plt.show()
            fig.savefig('thesis/machine_learning/graphs/graph_spin_'+str(N)+'_h_'+str(h)+'.png', bbox_inches='tight')
            fig.savefig('thesis/machine_learning/graphs/graph_spin_' + str(N) + '_h_' + str(h) + '.pdf', bbox_inches='tight')

            print("#########################################################")


