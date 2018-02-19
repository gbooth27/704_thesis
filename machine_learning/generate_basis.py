import numpy as np


class Psi(object):

    def __init__(self, n):
        self.size = n
        self.basis = self.generate()
        self.weights = self.gen_weights()

    def recusive_gen(self, state, index, S):
        """
        recursively generates all possible combos of state basis vectors
        :param state: initial state (all zeros)
        :param index: initial index to change
        :return:
        """
        # base case
        if index >= len(state):
            return S
        else:
            # append the current state
            S.append(state)
            # recursively generate without bit flip
            self.recusive_gen(list(state), index+1, S)
            # flip bit then recursively generate
            state[index] = 1
            self.recusive_gen(list(state), index + 1, S)
            return S

    def generate(self):
        """
        Generates a vector of all 2^n possible spin basis states of the system
        :return: numpy array of various states
        """
        S = []
        s_initial = [0 for _ in range(self.size)]
        S = self.recusive_gen(s_initial, 0, S)
        S = sorted(S)
        arr = np.array(S)
        return arr

    def gen_weights(self):
        """
        generates the vector of weights
        :return: psi representing weights
        """
        psi = [1 for _ in range(2**self.size)]
        return np.array(psi)

if __name__ == '__main__':
    p = Psi(4)
    print(p.basis)

