import copy

from swe_xz import *

def initial_condition(x, L, a):
    return a * np.cos(np.pi * x / L)

def analytical_freesurface(t, omega, a):
    return a * np.cos(omega*t)

def plot_accuracy(Dt, e2, e3, e4):
    fig = plt.figure(figsize=(7,7))
    plt.loglog(Dt, e2, 'r--')
    plt.loglog(Dt, e3, 'b--')
    plt.loglog(Dt, e4, 'k--')
    plt.loglog(Dt, e2, 'ro', label="u-velocity")
    plt.loglog(Dt, e3, 'bo', label="free surface")
    plt.loglog(Dt, e4, 'ko', label="pressure")
    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.ylabel("Error")
    plt.xlabel(r"$\Delta t$")
    plt.legend(loc='upper left', numpoints=1)
    return fig 

def timeseries_freesurface(t, T0, h_ana, h_c, h_p, a):
    fig = plt.figure(figsize=(11,5))
    t_ = t / T0
    plt.plot(t_, h_c/a, 'r-', lw=1.75, label="Correction")
    plt.plot(t_, h_p/a, 'b-', lw=1.75, label="Projection")
    plt.plot(t_, h_ana/a, 'k--', lw=1.75, label="Analytical")
    plt.ylabel('h/a')
    plt.xlabel('t/T0')
    plt.legend(loc="upper left")
    return fig 

def part5_timeseries(Dt_divisor=20):
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
    Dt = T / Dt_divisor
    ## vary timestep size
    ## create problem
    prob_c = SweXZProblem(
        L,
        D,
        t_max,
        N_i,
        N_k,
        Dt,
        theta,
        timeseries_loc=0,
        hydrostatic=False,
        pressure_method="correction",
    )
    ## initial condition and analytical solution
    prob_c.set_h_initial(initial_condition(prob_c.xc, L, a))
    run_swe_xz_problem(prob_c)
    ## prediction method
    prob_p = SweXZProblem(
        L,
        D,
        t_max,
        N_i,
        N_k,
        Dt,
        theta,
        timeseries_loc=0,
        hydrostatic=False,
        pressure_method="projection",
    )
    ## initial condition and analytical solution
    prob_p.set_h_initial(initial_condition(prob_p.xc, L, a))
    run_swe_xz_problem(prob_p)
    ## analytical free surface
    h_ana = analytical_freesurface(prob_c.t, omega, a)
    ## plot
    fig = timeseries_freesurface(prob_c.t, T, h_ana, prob_c.h_timeseries, prob_p.h_timeseries, a)
    fig.savefig('fig/timeseries_freesurface_projection_correction_{}.png'.format(Dt_divisor), dpi=300)

def part5(pressure_method='correction', rng=5):
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
    if pressure_method == 'projection':
        Dt_range = np.asarray([T / (10 * 2**(i+3)) for i in range(rng)])
    else:
        Dt_range = np.asarray([T / (10 * 2**(i+1)) for i in range(rng)])

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
            pressure_method=pressure_method,
        )
        ## initial condition and analytical solution
        prob.set_h_initial(initial_condition(prob.xc, L, a))
        ## run model
        run_swe_xz_problem(prob)
        ## save data to plot
        if j > 0:
            u_norm = np.linalg.norm(prob.u_out - u_old)
            u_list.append(u_norm)
            h_norm = np.linalg.norm(prob.h_out - h_old)
            h_list.append(h_norm)
            q_norm = np.linalg.norm(prob.q_out - q_old)
            q_list.append(q_norm)
        u_old = prob.u_out
        h_old = prob.h_out
        q_old = prob.q_out
        # fig = snapshot_velocity_freesurface(prob.xc, prob.zc, prob.u_out, 
        #                                     prob.w_out, prob.h_out, prob.L, prob.D)
    fig2 = plot_accuracy(Dt_range[1:], u_list, h_list, q_list)
    fig2.savefig("fig/plot_accuracy_{}.png".format(pressure_method), dpi=300)

if __name__ == '__main__':
    rng = 8
    # part5('correction', rng=rng)
    part5('projection', rng=rng)
    # part5_timeseries(Dt_divisor=20)
    # part5_timeseries(Dt_divisor=160)
    plt.show()