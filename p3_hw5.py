import numpy as np 

from swe_xz import *


def initial_condition(x, L, a):
    return a * np.cos(np.pi * x / L)

def analytical_freesurface(t, omega, a):
    return a * np.cos(omega*t)

def part3():
    ## parameters/settings
    N_i = 20
    N_k = 20
    L = 1
    D = 1
    k = np.pi / L # wave number = 2 pi / lambda (lambda = 2L)
    c = np.sqrt((9.81/k)*np.tanh(k*D))
    omega = c * k
    T = 2 * np.pi / (c*k)
    t_max = 2 * T
    Dt = T / 20
    theta = 0.5
    a = 0.01
    ## create problem
    prob = SweXZProblem(
        L,
        D,
        t_max,
        N_i,
        N_k,
        Dt,
        theta,
        timeseries_loc=0,
        hydrostatic=True,
    )
    prob.set_h_initial(initial_condition(prob.xc, L, a))
    ## run model
    run_swe_xz_problem(prob)
    fig = snapshot_velocity_freesurface(prob.xc, prob.zc, prob.u_out, 
                                        prob.w_out, prob.h_out, prob.L, prob.D)
    ## free surface timeseries plot
    h_analytical = analytical_freesurface(prob.t, omega, a)
    h_list = [h_analytical, prob.h_timeseries]
    labels = ["Analytical", "Non-hydrostatic, D/L = 1"]
    fig2 = timeseries_freesurface(prob.t, T, h_list, a, labels, [1,1])
    ## vertical velocity profiles plot
    u_list = []
    # fig3 = snapshot_velocity_profiles(prob.zc, D, )
    plt.show()

if __name__ == '__main__':
    part3()
