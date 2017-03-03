import copy

from swe_xz import *


def initial_condition(x, L, a):
    return a * np.cos(np.pi * x / L)

def plot_dispersion_relations(D_range, c_list_h, c_list_nh, c_list_t, c0, g=9.81):
    fig = plt.figure()
    plt.axhline(1, ls=':', color='k', lw=1.75, label='Deep-water limit')
    plt.plot(D_range, np.sqrt(D_range*g)/c0, 'k--', lw=1.75, label='Shallow-water limit')
    plt.plot(D_range, c_list_t/c0, 'k-', lw=1.75, label='Intermediate')
    plt.plot(D_range, c_list_h/c0, 'r-', lw=1.75, label='Hydrostatic')
    plt.plot(D_range, c_list_nh/c0, 'b-', lw=1.75, label='Non-hydrostatic')
    plt.xlabel("D/L")
    plt.ylabel("c/c0")
    plt.legend(loc="upper left")
    return fig 

def part4():
    ## const parameters/settings
    g = 9.81
    N_i = 20
    N_k = 20
    L = 1
    theta = 0.5
    a = 0.01
    ## lists of things to plot
    c_list_h = []
    c_list_nh = []
    c_list_t = []
    ## vary domain depth
    D_range = np.arange(0.1,1.1,0.1)
    for j, Dj in enumerate(D_range):
        for hydrostatic in (True, False):
            k = np.pi / L # wave number = 2 pi / lambda (lambda = 2L)
            c = np.sqrt((g/k)*np.tanh(k*Dj))
            c0 = np.sqrt(g/k)
            omega = c * k
            T = 2 * np.pi / (c*k)
            t_max = 2 * T
            Dt = T / 20
            ## create problem
            prob = SweXZProblem(
                L,
                Dj,
                t_max,
                N_i,
                N_k,
                Dt,
                theta,
                timeseries_loc=0,
                hydrostatic=hydrostatic,
            )
            ## initial condition and analytical solution
            prob.set_h_initial(initial_condition(prob.xc, L, a))
            ## run model
            run_swe_xz_problem(prob)
            ## store results
            if hydrostatic:
                c_list_h.append(0.5 * L / prob.t_change)
            else:
                c_list_nh.append(0.5 * L / prob.t_change)
                ## theoretical dispersion relation
                c_list_t.append(c)
    fig = plot_dispersion_relations(D_range, np.asarray(c_list_h), np.asarray(c_list_nh), 
                                    np.asarray(c_list_t), c0, g=g)
    fig.savefig('fig/plot_dispersion_relations.png', dpi=300)

if __name__ == '__main__':
    part4()