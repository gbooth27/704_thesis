import numpy as np
import scipy as sp
import thesis.hamiltonian.tfim as tfim
from scipy import linalg



class Psi(object):

    def __init__(self, n, h):
        self.size = n
        self.basis = self.generate()
        self.weights = self.gen_weights()
        self.normalize()
        self.Hamiltonian = self.get_ham(h)
        self.ground = []#self.get_ground()
        self.collapsed = self.collapse_on_axis()

    def min_energy(self, p):
        """
        get the min energy
        :param p:
        :return:
        """
        un_norm = np.dot(p.T, self.Hamiltonian.dot(p))
        norm = un_norm / np.dot(p.T, p)
        return norm

    def get_ham(self, h):
        """
        Generates the hamiltonian matrix for the given wavefunction.
        :param h: tuning parameter
        :return:
        """
        loaded_params, JZZ, ZZ, Mz, Ms = tfim.load_diag_ME("../hamiltonian/matrix")
        Mx = tfim.load_Mx("../hamiltonian/matrix")
        print(JZZ.shape)
        H = -JZZ - h*Mx
        return H

    def get_ground(self):
        """
        Generates the ground state of the system
        :return:
        """
        min = sp.optimize.minimize(self.min_energy, self.weights)
        self.ground = min.x

    def recursive_gen(self, state, index, S, state_dict):
        """
        Recursively generates all possible combos of state basis vectors.
        :param state: initial state (all zeros)
        :param index: initial index to change
        :return:
        """
        # base case
        if index >= len(state):
            return S
        else:
            # append the current state
            if str(state) not in state_dict:
                S.append(list(state))
                state_dict[str(state)] = True
            # recursively generate without bit flip
            self.recursive_gen(list(state), index+1, S, state_dict)
            # flip bit then recursively generate
            state[index] = 1
            # add if not already there
            if str(state) not in state_dict:
                S.append(list(state))
                state_dict[str(state)] = True

            self.recursive_gen(list(state), index + 1, S, state_dict)
            return S

    def generate(self):
        """
        Generates a vector of all 2^n possible spin basis states of the system.
        :return: numpy array of various states
        """
        S = []
        s_initial = [0 for _ in range(self.size)]
        S = self.recursive_gen(s_initial, 0, S, {})
        arr = np.array(S)
        for i in range(len(S)):
            S[i] = np.array(S[i]).reshape((len(S[i]), 1))
        return arr

    def gen_weights(self):
        """
        Generates the vector of weights.
        :return: psi representing weights
        """
        psi = [1 for _ in range(2**self.size)]
        return np.array(psi, dtype=np.longdouble).reshape((len((psi)), 1))

    def normalize(self):
        """
        normalize wavefunction
        :return:
        """
        self.weights = self.weights / np.sqrt(np.dot(self.weights.T, self.weights))

    def binary_to_int(self, state):
        """
        Takes a list form of a state and converts it's binary representation
        into a base 10 number.
        :param state: list of 0's and 1's
        :return: base 10 representation (eg. [0,1,0] -> 2)
        """
        string_state = ""
        for num in state:
            string_state += str(num)
        num = int(string_state, 2)
        return num

    def collapse_on_axis(self):
        """
        collapses the hamiltonain matrix on each axis and sums them
        :return:
        """
        ham = self.Hamiltonian.toarray()
        column_wise = np.sum(ham, axis=0)
        row_wise = np.sum(ham, axis=1)
        end = np.add(column_wise, row_wise)
        return end

    def diag(self):
        """
        diagonalizes hamiltonian then returns the ground state energy of
        the system
        :return: ground state energy
        """
        E, v = linalg.eigh(self.Hamiltonian.todense())
        E = sorted(E)
        return E[0]



if __name__ == '__main__':
    p = Psi(6, 1)
    print(p.basis)
    print(p.Hamiltonian.shape)
    print(p.binary_to_int(p.basis[2]))
    vec = np.dot(np.dot(p.weights.T, p.Hamiltonian.toarray()), p.weights)
    print(vec[0][0])
    print(vec[0][0]/np.dot(p.weights.T, p.weights)[0][0])
    print(p.weights.shape)
