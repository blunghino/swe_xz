import copy

from swe_xz import *

def initial_condition(x, L, a):
    return a * np.cos(np.pi * x / L)

def analytical_freesurface(t, omega, a):
    return a * np.cos(omega*t)

def part5():
    ## const parameters/settings
    g = 9.81
    N_i = 20
    N_k = 20
    L = 1
    D = 1 
    theta = 0.5
    a = 0.01
    k = np.pi / L # wave number = 2 pi / lambda (lambda = 2L)
    c = np.sqrt((g/k)*np.tanh(k*D))
    omega = c * k
    T = 2 * np.pi / (c*k)
    t_max = 10 * T
    Dt_range = np.asarray([T / (10 * 2**(i+1)) for i in range(3)])
    ## analytical free surface
    h_analytical = analytical_freesurface(np.arange(0, t_max+Dt_range[0], Dt_range[0]), omega, a)
    ## lists of things to plot
    u_list = []
    h_list = []
    q_list = []
    ## vary timestep size
    for j, Dt in enumerate(Dt_range):
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
            hydrostatic=False,
        )
        ## initial condition and analytical solution
        prob.set_h_initial(initial_condition(prob.xc, L, a))
        ## run model
        run_swe_xz_problem(prob)
        if j > 0:
            u_norm = np.linalg.norm(prob.u_out - u_old)
            u_list.append(u_norm)
            u_old = prob.u_out
        else:
            u_old = prob.u_out
    

if __name__ == '__main__':
    part5()