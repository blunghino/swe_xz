import copy

from swe_xz import *


def initial_condition(x, L, a):
    return a * np.cos(np.pi * x / L)

def analytical_freesurface(t, omega, a):
    return a * np.cos(omega*t)

def part3():
    ## const parameters/settings
    N_i = 20
    N_k = 20
    D = 1
    theta = 0.5
    a = 0.01
    ## lists of things to plot
    h_list = []
    h_analytical_list = []
    u_list = []
    w_list = []
    ## labels for plots
    labels = 3 * ["Non-hydrostatic", "Hydrostatic"]
    subplots = [1,1,2,2,3,3]
    ## vary domain length
    for j, Lj in enumerate([1, 1, 4, 4, 8, 8]):
        hydrostatic = j % 2
        k = np.pi / Lj # wave number = 2 pi / lambda (lambda = 2L)
        c = np.sqrt((9.81/k)*np.tanh(k*D))
        omega = c * k
        T = 2 * np.pi / (c*k)
        t_max = 2 * T
        Dt = T / 20
        ## create problem
        prob = SweXZProblem(
            Lj,
            D,
            t_max,
            N_i,
            N_k,
            Dt,
            theta,
            timeseries_loc=0,
            hydrostatic=hydrostatic,
        )
        ## initial condition and analytical solution
        prob.set_h_initial(initial_condition(prob.xc, Lj, a))
        if hydrostatic:
            h_analytical_list.append(analytical_freesurface(prob.t, omega, a))
        ## run model
        run_swe_xz_problem(prob)
        ## store results
        h_list.append(prob.h_timeseries)
        u_list.append(prob.u_out)
        w_list.append(prob.w_out)
        # fig = snapshot_velocity_freesurface(prob.xc, prob.zc, prob.u_out, 
        #                                     prob.w_out, prob.h_out, prob.L, prob.D)
        # fig.savefig("fig/snapshot_velocity_freesurface_{}_L-{}.png".format(labels[j], Lj))
    ## free surface timeseries plot
    fig2 = timeseries_freesurface(prob.t, T, h_list, h_analytical_list, a, labels, subplots)
    fig2.savefig("fig/timeseries_freesurface.png", dpi=300)
    ## vertical velocity profiles plot
    fig3 = snapshot_velocity_profiles(prob.zc, D, prob.Dz, u_list, w_list, labels, subplots)
    plt.savefig("fig/snapshot_velocity_profiles.png", dpi=300)

if __name__ == '__main__':
    part3()
