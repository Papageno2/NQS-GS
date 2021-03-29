# encoding: utf-8

import numpy as np
from operators.HS_spin2d import Heisenberg2DTriangle, Heisenberg2DSquare, value2onehot
# from operators.HS_spin2d import value2onehot
from operators.tfim_spin2d import TFIMSpin2D
from utils import decimalToAny
import multiprocessing
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

def create_basis(Dp, L, W, ms):
    N = L*W
    basis_index = []
    basis_state = []
    for i in range(Dp**N):
        num_list = decimalToAny(i,Dp)
        state_v = np.array([0]*(N - len(num_list)) + num_list)
        if (state_v - (Dp-1)/2).sum() == ms:
            basis_index.append(i)
            basis_state.append(state_v.reshape(L,W))
        '''
        state_v = np.array([0]*(N - len(num_list)) + num_list)
        basis_index.append(i)
        basis_state.append(state_v.reshape(L,W))
        '''
    return basis_index, basis_state

def onehot2value(state, Dp): 
    state_v = np.arange(0,Dp).reshape(Dp,1,1)*np.squeeze(state)
    return np.sum(state_v,0).astype(dtype=np.int8)

def find_matrix_element(ham, basis_index, state, Dp, pow_list):
    ham_list = np.zeros(len(basis_index))
    ustates, ucoeffs = ham.find_states(value2onehot(state, Dp))
    for ustate, ucoeff in zip(ustates, ucoeffs):
        ustate_v = onehot2value(ustate, Dp).reshape(-1)
        if ustate_v.sum() != 0: 
            num = np.sum(pow(2,pow_list)*ustate_v)
            j = basis_index.index(num)
            ham_list[j] += ucoeff
    return ham_list

def main(L, W, Dp, ms):
    state_size = [L,W,Dp]
    _ham = Heisenberg2DSquare(state_size=state_size, pbc=False)
    basis_index, basis_state = create_basis(Dp, L, W, ms)
    ham_matrix = np.zeros([len(basis_index), len(basis_index)])
    pow_list = np.arange(L*W-1, -1, -1)

    pool = multiprocessing.Pool(4)
    results = []
    for state in basis_state:
        results.append(pool.apply_async(find_matrix_element, 
                (_ham, basis_index, state, Dp, pow_list, )))
    pool.close()
    pool.join()

    cnt = 0
    for res in results:
        ham_matrix[cnt] = res.get()
        cnt += 1

    # print(ham_matrix)
    # ham_matrix = csr_matrix(ham_matrix)
    # D = eigs(ham_matrix, k=2, which='SR', return_eigenvectors=False)
    # print(D)
    # D_sorted = np.sort(D)
    D, V = np.linalg.eigh(ham_matrix)
    V_sorted = V[:, D.argsort()]
    g_state= V_sorted[:,0]
    D_sorted = np.sort(D)
    g_energy = D_sorted[0]
    # print(D_sorted[1]/L/W)
    # print(D_sorted[2]/L/W)
    return g_energy, g_state, ham_matrix
    # return D_sorted[1] - D_sorted[0]

if __name__ == '__main__':
    L = 4
    W = 4
    ms = 0 if L*W%2 == 0 else -0.5

    basis_index, basis_state = create_basis(2,L,W,ms)
    print(len(basis_index))
    '''
    ge = []
    # D, V, H = main(0,L,W,2,ms)
    for g in np.arange(0,0.5,0.005):
       ge.append(main(g,L,W,2,ms))
    print(ge)
    '''
    # print(np.arange(10,-1,-1))
    ge, gs, H = main(L,W,2,ms)
    print(ge/L/W)
    # sio.savemat('./data/ed_data_HS2Dka_L4W3.mat',{'ge':ge,'gs':gs,'H':H})
    #plt.figure()
    #plt.plot(np.arange(0,0.5,0.005),ge)
    #plt.show()
