import numpy as np 

from swe_xz import *


def initial_condition(x, L, a=0.01):
    return a * np.cos(np.pi * x / L)

def part3():
    ## parameters/settings
    N_i = 20
    N_k = 20
    L = 1
    D = 1
    k = np.pi / L # wave number = 2 pi / lambda (lambda = 2L)
    c = np.sqrt((9.81/k)*np.tanh(k*D))
    T = 2 * np.pi / (c*k)
    t_max = 2 * T
    Dt = T / 20
    theta = 0.5
    ## create problem
    prob = SweXZProblem(
        L,
        D,
        t_max,
        N_i,
        N_k,
        Dt,
        theta,
        hydrostatic=False,
    )
    prob.set_h_initial(initial_condition(prob.xc, L))
    ## run model
    run_swe_xz_problem(prob)
    fig = snapshot_velocity_freesurface(prob.xc, prob.zc, prob.u_out, 
                                        prob.w_out, prob.h_out, prob.D)
    plt.show()

if __name__ == '__main__':
    part3()
