import copy

from swe_xz import *


def initial_condition(x, L, a):
    return a * np.cos(np.pi * x / L)

def analytical_freesurface(t, omega, a):
    return a * np.cos(omega*t)

def timeseries_freesurface(t, T0, h_list, h_analytical_list, a, labels, subplots):
    fig = plt.figure(figsize=(13,9))
    n_sp = max(subplots)
    plt.subplot(n_sp, 1, 1)
    plt.title("D/L = 1")
    plt.subplot(n_sp, 1, 2)
    plt.title("D/L = 1/4")
    plt.subplot(n_sp, 1, 3)
    plt.title("D/L = 1/8")
    plt.xlabel('t/T0')
    t_ = t / T0
    ## first entry in h_list should be analytical, plotted on every subplot
    for j, h in enumerate(h_list):
        plt.subplot(n_sp, 1, subplots[j])
        if not j % 2:
            plt.plot(t_, h_analytical_list[j//2]/a, ls='--', c='k', label="Analytical")
        plt.plot(t_, h/a, label=labels[j])
        plt.ylabel('h/a')
    plt.legend(loc="upper left")
    return fig 

def snapshot_velocity_profiles(zc, D, Dz, u_list, w_list, labels, subplots):
    zc_ = zc / D
    zf_ = np.arange(0, D + Dz, Dz) / D 
    N_i = w_list[0].shape[0]
    mid = N_i // 2 
    n_sp = 3
    fig = plt.figure(figsize=(13,9))
    plt.subplot(1, n_sp, 1)
    plt.title("D/L = 1")
    plt.subplot(1, n_sp, 2)
    plt.title("D/L = 1/4")
    plt.subplot(1, n_sp, 3)
    plt.title("D/L = 1/8")
    for j, (u, w) in enumerate(zip(u_list, w_list)):
        u_ck = 0.5 * (u[mid-1,:] + u[mid,:])
        w_rk = 0.5 * (3 * w[-1,:] - w[-2,:])
        u_ = np.abs(u_ck) / np.max(np.abs(u_ck))
        w_ = np.abs(w_rk) / np.max(np.abs(w_rk))
        plt.subplot(1, n_sp, subplots[j])
        ## different symbol between hydrostatic/non-hydrostatic
        if labels[j][:3] == 'Non':
            ls = '--'
        else:
            ls = '-'
        plt.plot(u_, zc_, 'r', ls=ls, lw=1.5, label="u, {}".format(labels[j]))
        plt.plot(w_, zf_, 'k', ls=ls, lw=1.5, label="w, {}".format(labels[j]))
        plt.xlabel('Normalized Velocity')
        plt.ylabel('z/D')
        plt.xlim(right=1.1)
    plt.legend(loc='upper left')
    return fig 

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
