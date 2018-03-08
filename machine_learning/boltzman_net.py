import rbm.rbm as rbm


def run_rbm(psi):
    """
    Sets up and runs an RBM based on wavefunction PSI
    :param psi: wavefunction
    :return:
    """
    # Create the RBM with N visible and 2N hidden
    net = rbm.RBM(num_visible=psi.size, num_hidden=2*psi.size)
