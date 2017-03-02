"""
unit tests for functions in swe_xz.py
"""
from swe_xz import *


def test_construct_LHS_pressure():
    csr = construct_LHS_pressure(2,2,3,3)
    A = csr.toarray()
    print(np.diag(A))

def test_calc_S():
    N_i = 3
    N_k = 3
    u = np.zeros((N_i+1,N_k))
    h = np.zeros(N_i)
    q = np.zeros((N_i, N_k))
    S = calc_S(u, h, q, 1, 1, 0.5, 9.81)

def test_calc_RHS_freesurface():
    N_i = 3
    N_k = 6
    u = np.zeros((N_i+1,N_k))
    S = np.zeros_like(u)
    h = np.zeros(N_i)
    R = calc_RHS_freesurface(u, S, h, 1, 1, 1, 0.5)

if __name__ == '__main__':
    # test_calc_RHS_freesurface()
    test_construct_LHS_pressure()
