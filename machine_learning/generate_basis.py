import numpy as np
import hamiltonian.tfim as tfim

class Psi(object):

    def __init__(self, n, h):
        self.size = n
        self.basis = self.generate()
        self.weights = self.gen_weights()
        self.Hamiltonian = self.get_ham(h)

    def get_ham(self, h):
        """
        Generates the hamiltonian matrix for the given wavefunction.
        :param h: tuning parameter
        :return:
        """
        loaded_params, JZZ, ZZ, Mz, Ms = tfim.load_diag_ME(self.size)
        Mx = tfim.load_Mx(self.size)
        H = -JZZ - h*Mx
        return H

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
        return arr

    def gen_weights(self):
        """
        Generates the vector of weights.
        :return: psi representing weights
        """
        psi = [1 for _ in range(2**self.size)]
        return np.array(psi)

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


if __name__ == '__main__':
    p = Psi(4, 1)
    print(p.basis)
    print(p.Hamiltonian)
    print(p.binary_to_int(p.basis[2]))

