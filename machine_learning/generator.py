import numpy as np

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

