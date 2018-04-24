#!/usr/bin/env python

""""tfim_diag.py
    Chris Herdman
    06.07.2017
    --Exact diagonalization for transverse field Ising models
    --Requires: tfim.py, numpy, scipy.sparse, scipy.linalg, progressbar
"""
import hamiltonian.tfim as tfim
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy import linalg
import argparse

###############################################################################
def main():
    
    # Parse command line arguements
    ###################################
    parser = argparse.ArgumentParser(description=(
                            "Exact numerical diagonalization of "
                            "transverse field Ising Models of the form:\n"
                            "H = -\sum_{ij} J_{ij}\sigma^z_i \sigma^z_j" 
                                                    "- h \sum_i \sigma^x_i") )
    parser.add_argument('lattice_specifier', 
                            help=(  "Either: L (linear dimensions of the system)"
                                    " or the filename base of matrix files") )
    parser.add_argument('-D', type=int,default=1,
                                        help='Number of spatial dimensions')
    parser.add_argument('--obc',action='store_true',
                            help='Open boundary condintions (deault is PBC)')            
    parser.add_argument('--h_min', type=float, default=0.0,
                            help='Minimum value of the transverse field')
    parser.add_argument('--h_max', type=float, default=4.0,
                            help='Maximum value of the transverse field')    
    parser.add_argument('--dh', type=float, default=0.5,
                            help='Tranverse fied step size')
    parser.add_argument('-J', type=float, default=1.0,
                            help='Nearest neighbor Ising coupling')
    parser.add_argument('-k', type=int,default=3,
                                        help='Number eigenvalues to resolve')
    parser.add_argument('-o', default='output', help='output filename base')                                        
    parser.add_argument('--full',action='store_true',
                            help='Full (rather than Lanczos) diagonalization')
    parser.add_argument('--save_state',action='store_true',
                            help='Save ground state to file')
    parser.add_argument('--init_v0',action='store_true',
                            help='Start Lanzcos with previous ground state')    
    parser.add_argument('--load', action='store_true',
                                            help='Load matrices from file' )
    parser.add_argument('--fidelity', action='store_true',
                                            help='Compute fidelities' )
    parser.add_argument('--delta_h_F0', type=float, default = 1E-4,
                                        help='Inital \Delta h for fidelity' ) 
    parser.add_argument('--N_F_steps', type=int, default = 3,
                                    help='Number of steps for fidelity' )                                                                
    parser.add_argument('--overlap', action='store_true',
                                    help='Compute the overlap distribution' )
    parser.add_argument('--N_ovlp_samples', type=int, default = 10**4,
                        help='Number of samples of the overlap distribution' )
    parser.add_argument('--SK', action='store_true',
                        help='SK model with infinite range ZZ interactions' )
            
    args = parser.parse_args()
    ###################################
    
    # Load matricies from file
    ###################################
    load_matrices = args.load
    if load_matrices:
        loaded_params, JZZ, ZZ, Mz, Ms = tfim.load_diag_ME(
                                                    args.lattice_specifier)
        Mx = tfim.load_Mx(args.lattice_specifier)
    ###################################
    
    # Set calculation Parameters
    ###################################
    out_filename = args.o + '.dat'
    if load_matrices:
        L = loaded_params['L']
        D = len(L)
        PBC = loaded_params['PBC']
        J = loaded_params['J']
    else:
        D = args.D
        L = [ int(args.lattice_specifier) for d in range(D) ]
        PBC = not args.obc
        J = args.J
    k = args.k
    init_v0 = args.init_v0
    full_diag = args.full
    SK = args.SK
    save_state = args.save_state
    if save_state:
        state_filename = args.o + '_psi0.dat'
    
    fidelity_on = args.fidelity
    if fidelity_on:
        delta_h_F0 = args.delta_h_F0
        N_F_steps = args.N_F_steps
        dhf = np.flip(delta_h_F0/(2**(np.arange(N_F_steps))),axis=0)
        F2 = np.zeros(dhf.shape)
        F2_filename = args.o + '_F2.dat'
    
    overlap_on = args.overlap
    if overlap_on:
        N_ovlp_samples = args.N_ovlp_samples
        Pq_filename = args.o + '_Pq.dat'
    
    h_arr = np.arange(args.h_min,args.h_max+args.dh/2,args.dh)
    parameter_string = ("D = {}, L = {}, PBC = {}, J = {},"
                        " k = {}".format(D, L, PBC, J, k) )
    print('\tStarting tfim_diag using parameters:\t' + parameter_string)   
    ###################################
    
    # Setup physical quantities
    ##################################
    # Quantities to write ouput file    
    phys_keys = ['h', 'e0', 'Delta_1', 'Delta_2', 'Mx', 'Mz2', 'Cnn', 'Ms2'] 
    phys = {}           # Dictionary for values
    ##################################
    
    # Build lattice and basis
    ###################################
    lattice = tfim.Lattice(L, PBC)
    N = lattice.N
    basis = tfim.IsingBasis(lattice)
    ###################################
    
    # Setup output data files
    ##################################
    width = 25
    precision = 16
    header_list = [tfim.phys_labels[key] for key in phys_keys]
    header = ''.join(['{:>{width}}'.format(head,width=width) 
                                            for head in header_list])
    out_file = open(out_filename, 'w')
    print( "\tData will write to {}".format(out_filename) )
    out_file.write( '#\ttfim_diag parameters:\t' + parameter_string + '\n' 
                    + '#' + header[1:] + '\n' )
    
    if save_state:
        state_file = open(state_filename, 'w')
        print( "\tGround state will write to {}".format(state_filename) )
        state_file.write( 
                "# tfim_diag parameters:\t{}\n".format(parameter_string)
                + "#{:>{width_h}}{:>{width_psi}}\n".format( 'h', '\psi_0' , 
                        width_h= ( width - 1 ), width_psi=(width +1 ) )      )
    
    if fidelity_on:
        F2_header =( "#{:>{width}}".format( 'h', width=(width - 1) )
                    + ''.join(['{:{width}.{prec}e}'.format(dhfi,
                    width=(width+1), prec=(precision-1)) for dhfi in dhf] ) )
        F2_file = open(F2_filename, 'w')
        print( "\tFidelities will write to {}".format(F2_filename) )
        F2_file.write( '#\ttfim_diag parameters:\t' + parameter_string + '\n' 
                        + '#' + F2_header[1:] + '\n' )
                        
    if overlap_on:
        q = np.arange(-N,N+1,2)/float(N)
        Pq_header =( "#{:>{width}}".format( 'h', width=(width - 1) )
                    + ''.join(['{:{width}.{prec}e}{:>{width}}'.format(qi,
                                'error', width=(width+1), prec=(precision-1) 
                                                        ) for qi in q] ) )
        Pq_file = open(Pq_filename, 'w')
        print( "\tOverlap distributions will write to {}".format(Pq_filename) )
        Pq_file.write( '#\ttfim_diag parameters:\t' + parameter_string + '\n' 
                        + '#' + Pq_header[1:] + '\n' )
    ##################################
    
    # Build Matricies
    ###################################
    if not load_matrices:
        print( '\tBuilding matrices...' )
        JZZ, ZZ = tfim.z_correlations_NN(lattice,basis,J)
        Mz, Ms = tfim.z_magnetizations(lattice,basis)
        Mx = tfim.build_Mx(lattice,basis)
        
        if SK:
            Jij = tfim.Jij_instance(N,J)
            #Jij = np.ones((N/2,N))/N
            JZZ = tfim.JZZ_SK(basis,Jij)
    ###################################
    
    
    # Main Diagonalization Loop
    #######################################################
    if full_diag:
        print("\tStarting full diagaonalization with h in ({},{}), "
                                "dh = {}".format(h_arr[0], h_arr[-1],args.dh) )
    else:
        print("\tStarting sparse diagaonalization with k={} and "
                "h in ({},{}), dh ={}".format(k,h_arr[0], h_arr[-1],args.dh) )
    v0 = None
    for h in h_arr:
        
        H = -JZZ - h*Mx    
        if full_diag:
            # Full diagonalize
            E,v = linalg.eigh(H.todense())
        else:
            # Sparse diagonalize
            E,v = spla.eigsh(H, k=k, which='SA', v0=v0)
        
        # Sort eigenvalues/vectors
        sort_order = np.argsort(E)
        E = E[sort_order]
        v = v[:,sort_order]
        
        # Grab Energies & ground state
        e0 = E[0]/N
        Delta = E - E[0]
        psi0 = v[:,0]
        
        # Set starting vector for Lanczos:
        if not full_diag and init_v0:
            v0 = psi0
                
        # Compute expectation values
        ###################################
        Mx0 = np.real((psi0.conj().T).dot(Mx.dot(psi0)))/N
        Mz20 = np.real((psi0.conj().T).dot((Mz.power(2)).dot(psi0)))/(N**2)
        Cnn = np.real((psi0.conj().T).dot(ZZ.dot(psi0)))/lattice.N_links
        Ms20 = np.real((psi0.conj().T).dot((Ms.power(2)).dot(psi0)))/(N**2)
        ###################################
        
        # Compute fidelities
        ###################################
        if fidelity_on:            
            for i, dhfi in enumerate(dhf):
                H_F = -JZZ - (h+dhfi)*Mx 
                E_F,v_F = spla.eigsh(H_F, k=2, which='SA', v0=psi0)
                # Sort eigenvalues/vectors
                sort_order_F = np.argsort(E_F)
                E_F = E_F[sort_order_F]
                v_F = v_F[:,sort_order_F]
                F2[i] = (np.absolute(np.vdot(v_F[:,0], psi0)))**2
        ###################################
    
        # Overlap distribution
        ###################################
        if overlap_on:
            Pq,Pq_err,q = basis.sample_overlap_distribution(psi0,N_ovlp_samples)
        ###################################
        
        # Put physical values in phys dictionary
        ###################################
        phys['h'] = h
        phys['e0'] = e0
        phys['Delta_1'] = Delta[1]
        phys['Delta_2'] = Delta[2]
        phys['Mx'] = Mx0
        phys['Mz2'] = Mz20
        phys['Cnn'] = Cnn
        phys['Ms2'] = Ms20
        ###################################
        
        # Write data to output files
        ###################################
        data_list = [phys[key] for key in phys_keys]
        data_line = ''.join(['{:{width}.{prec}e}'.format(data,width=width,
                                    prec=precision) for data in data_list])
        out_file.write(data_line+ '\n')
                                
        # Write psi0 to file
        if save_state:
            np.savetxt(state_file, 
                        np.concatenate(([h],psi0)).reshape((1,psi0.shape[0]+1)), 
                        fmt='%{}.{}e'.format(width,precision-1) )
                        
        # Write fidelities to file
        if fidelity_on:
            np.savetxt(F2_file, 
                        np.concatenate(([h],F2)).reshape((1,F2.shape[0]+1)), 
                        fmt='%{}.{}e'.format(width,precision-1) )
                        
        # Write overlap distribution to file
        if overlap_on:
            Pq_line = np.zeros(1+2*len(Pq))
            Pq_line[0] = h
            Pq_line[1::2] = Pq
            Pq_line[2::2] = Pq_err
            np.savetxt(Pq_file, Pq_line.reshape((1,Pq_line.shape[0])), 
                                    fmt='%{}.{}e'.format(width,precision-1) )
        
    #######################################################
    
    # Close files
    out_file.close()
    if save_state:
        state_file.close()
    if fidelity_on:
        F2_file.close()
    if overlap_on:
        Pq_file.close()
    
if __name__ == "__main__":
    main()
