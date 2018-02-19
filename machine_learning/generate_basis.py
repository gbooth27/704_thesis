import numpy as np


class Psi(object):

    def __init__(self, n):
        self.size = n
        self.basis = self.generate()

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
            S.append(state)
            self.recusive_gen(list(state), index+1, S)
            state[index] = 1
            #S.append(state)
            self.recusive_gen(list(state), index + 1, S)
            return S



    def generate(self):
        """
        Generates a vector of all 2^n possible spin basis states of the system
        :param n: number of particles
        :return:
        """
        S = []
        s_initial = [0 for _ in range(self.size)]
        S = self.recusive_gen(s_initial, 0, S)
        arr = np.array(S)
        return arr

if __name__ == '__main__':
    p = Psi(4)
    print(p.basis)

