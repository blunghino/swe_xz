"""
solve the 2D-V shallow water equations
"""
import warnings

import numpy as np 
from numpy.matlib import repmat 
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse.linalg
from matplotlib import pyplot as plt 


class SweXZProblem:
    """
    class to store parameters for swe_xz model run
    """
    def __init__(self, L, D, t_max, N_i, N_k, Dt, theta=0.5,
                 hydrostatic=True, h_initial=None, g=9.81,
                 timeseries_loc=0):
        """
        set parameter values
        """
        ## domain length [m]
        self.L = L
        ## domain depth [m]
        self.D = D
        ## final time [s]
        self.t_max = t_max 
        ## number of grid cells in horizontal
        self.N_i = N_i
        ## number of grid cells in vertical
        self.N_k = N_k
        ## timestep size [s]
        self.Dt = Dt 
        ## x grid cell size [m]
        self.Dx = L / N_i
        ## z grid cell size [m]
        self.Dz = D / N_k
        ## number of timesteps
        self.N_t = t_max / Dt 
        ## cell centered grid x, z
        self.xc = np.arange(self.Dx/2, L + self.Dx/2, self.Dx)
        self.zc = np.arange(self.Dz/2, D + self.Dz/2, self.Dz)
        ## timesteps
        self.t = np.arange(0, t_max+Dt, Dt)
        ## bathymetry
        self.d = D * np.ones_like(self.xc)
        ## is the problem hydrostatic?
        self.hydrostatic = hydrostatic
        ## gravitational accleration [m/s^2]
        self.g = g 
        ## constant for theta method dicretization
        self.theta = theta 
        ## free surface initial condition
        if h_initial is None:
            ## vector of same length as cell centered x grid
            self.h_initial = np.zeros_like(self.xc)
        else:
            self.h_initial = h_initial
        ## data output
        self.h_timeseries = np.zeros_like(self.t)
        self.timeseries_loc = timeseries_loc
        self.timeseries_idx = self.xc.searchsorted(timeseries_loc)
        ## initialize vectors to store results
        self.h_out = np.zeros_like(self.h_initial)
        self.u_out = np.zeros((self.N_i+1, self.N_k))
        self.w_out = np.zeros((self.N_i, self.N_k+1))

    def set_h_initial(self, h_initial):
        self.h_initial = h_initial

    def calc_timeseries_val(self, h):
        """
        method to set value to store
        """
        if self.timeseries_loc == 0:
            return 1.5 * h[0] - 0.5 * h[1]
        elif self.timeseries_loc == self.L:
            return 1.5 * h[-1] - 0.5 * h[-2]
        else:
            return h[self.timeseries_idx]

def calc_S(u, h, q, Dt, Dx, theta, g):
    """
    S is an N_i+1 by N_k matrix containing explicit data
    for the u velocity field update 
    """
    S = np.zeros_like(u)
    N_k = S.shape[1]
    N_i = h.shape[0]
    h_mat = repmat(h.reshape(N_i,1), 1, N_k)
    S[1:-1,:] = u[1:-1,:] \
                - (g * Dt * (1-theta) / Dx) * (h_mat[1:,:] - h_mat[:-1,:]) \
                - (Dt / Dx) * (q[1:,:] - q[:-1,:])
    return S

def calc_RHS_freesurface(u, S, h, Dt, Dx, Dz, theta):
    """
    R is a length N_i vector containing the RHS for the
    free surface height update
    """
    return h - ((1-theta)*Dt*Dz/Dx) * np.sum(u[1:,:]-u[:-1,:], 1) \
             - (theta*Dt*Dz/Dx) * np.sum(S[1:,:]-S[:-1,:],1)

def construct_LHS_freesurface(N_i, D, Dt, Dx, theta, g):
    """
    L is a matrix of shape N_i x N_i  
    """
    L = np.zeros((N_i,N_i))
    a = theta**2 * g * D * Dt**2 / Dx**2
    L[1:-1,1:-1] = -a*np.eye(N_i-2, k=1) + (1+2*a)*np.eye(N_i-2) - a*np.eye(N_i-2, k=-1)
    ## fill in corners
    L[0,0] = 1 + a
    L[0,1] = -a
    L[1,0] = -a
    L[-1,-1] = 1 + a
    L[-1,-2] = -a
    L[-2,-1] = -a
    return csr_matrix(L) 

def predict_u_velocity(h_new, S, Dt, Dx, theta, g):
    """
    calculate the predicted u velocity before the pressure update
    if modeling a hydrostatic system this is used as the u velocity update 
    """
    N_i = h_new.shape[0]
    N_k = S.shape[1]
    u = np.zeros((N_i+1,N_k))
    h_mat = repmat(h_new.reshape(N_i,1), 1, N_k)
    u[1:-1,:] = S[1:-1,:] - (g*theta*Dt/Dx) * (h_mat[1:,:]-h_mat[:-1,:])
    return u 

def predict_w_velocity(w, q, Dt, Dz):
    """
    calculate the predicted w velocity using the n-1/2 pressure field 
    """
    w_str = np.zeros_like(w)
    w_str[:,1:-1] = w[:,1:-1] - (Dt/Dz) * (q[:,1:] - q[:,:-1])
    ## free surface boundary condition
    w_str[:,-1] = w[:,-1] + (2*Dt/Dz)*q[:,-1]
    return w_str

def calc_RHS_pressure(u_str, w_str, Dt, Dx, Dz):
    """
    return b a vector of length N_i x N_k that is the RHS for the 
    laplace equation for the non-hydrostatic pressure correction
    """
    b = ((u_str[1:,:] - u_str[:-1,:]) / Dx + (w_str[:,1:] - w_str[:,:-1]) / Dz) / Dt
    return b.flatten()

def construct_LHS_pressure(Dx, Dz, N_i, N_k):
    """
    returns a scipy.sparse.csr_matrix

    construct the Laplacian operator used to solve for the 
    non-hydrostatic pressure correction 
    """
    ## common values - eg One Over Dx Squared
    ooDx2 = 1 / Dx**2
    ooDz2 = 1 / Dz**2
    diag = -2 * (ooDx2 + ooDz2)
    ## dimension of a row/col
    n_q = N_i * N_k
    ## vectors to fill
    val = np.zeros(5*n_q, dtype=float)
    row = np.zeros(5*n_q, dtype=int)
    col = np.zeros(5*n_q, dtype=int)
    ctr = 0
    for i in range(N_i):
        for k in range(N_k):
            idx = i * N_k + k 
            if k == N_k-1:
                ## free surface corners
                if i == 0 or i == N_i-1:
                    row[ctr] = idx
                    col[ctr] = idx
                    val[ctr] = -ooDx2 - 3 * ooDz2
                    ctr += 1
                ## free surface no corner
                else:
                    row[ctr] = idx
                    col[ctr] = idx
                    val[ctr] = -2 * ooDx2 - 3 * ooDz2
                    ctr += 1 
            ## side walls
            elif i == 0 or i == N_i-1:
                ## corners with bottom
                if k == 0:
                    row[ctr] = idx
                    col[ctr] = idx
                    val[ctr] = - ooDx2 - ooDz2
                    ctr += 1
                ## side wall no corner
                else:
                    row[ctr] = idx
                    col[ctr] = idx
                    val[ctr] = -ooDx2 - 2 * ooDz2
                    ctr += 1
            ## bottom no corner
            elif k == 0:
                row[ctr] = idx
                col[ctr] = idx
                val[ctr] = -2 * ooDx2 - ooDz2
                ctr += 1                
            ## no boundary
            else:
                row[ctr] = idx
                col[ctr] = idx
                val[ctr] = diag
                ctr += 1
            if i != 0:
                row[ctr] = idx
                col[ctr] = idx - N_k
                val[ctr] = ooDx2
                ctr += 1
            if i != N_i-1:
                row[ctr] = idx
                col[ctr] = idx + N_k 
                val[ctr] = ooDx2
                ctr += 1
            if k != 0:
                row[ctr] = idx 
                col[ctr] = idx - 1
                val[ctr] = ooDz2
                ctr += 1
            if k != N_k-1:
                row[ctr] = idx
                col[ctr] = idx + 1
                val[ctr] = ooDz2
                ctr += 1
    return coo_matrix((val, (row,col)), shape=(n_q,n_q)).tocsr()

def update_u_velocity(u_str, q_c, Dt, Dx):
    """
    use the predicted u-velocity field and non-hydrostatic pressure 
    correction to update the u-velocity at time n+1
    """
    u = np.zeros_like(u_str)
    u[1:-1,:] = u_str[1:-1,:] - (Dt/Dx) * (q_c[1:,:]-q_c[:-1,:])
    return u 

def update_w_velocity(u, Dx, Dz):
    """
    update the w velocity field using the divergence 
    free property of the flow
    update by back substitution
    w[:,0] = 0
    """
    m, n = u.shape
    w = np.zeros((m-1, n+1))
    for k in range(0, n):
        w[:,k+1] = w[:,k] - (Dz/Dx) * (u[1:,k]-u[:-1,k])
    return w

def run_swe_xz_problem(p, cg_tol=1e-10):
    """
    steps to compute solution to `p`, a SweXZProblem

    all vectorized grids are stored row major 
    """
    ## number of points to store for u and w velocity fields
    n_fx = (p.N_i + 1) * p.N_k
    n_fz = p.N_i * (p.N_k + 1)
    ## number of grid points to store for q non-hydrostatic pressure field
    n_c = p.N_i * p.N_k 
    ## initialize vectors
    h = p.h_initial.copy()
    u = np.zeros((p.N_i+1, p.N_k))
    w = np.zeros((p.N_i, p.N_k+1))
    q = np.zeros((p.N_i, p.N_k))
    u_str = np.zeros_like(u)
    w_str = np.zeros_like(w)
    q_c = np.zeros_like(q)
    S = np.zeros_like(u) 
    ## these sparse matrices don't vary in time
    LHS_h = construct_LHS_freesurface(p.N_i, p.D, p.Dt, p.Dx, p.theta, p.g)
    LHS_q = construct_LHS_pressure(p.Dx, p.Dz, p.N_i, p.N_k)
    ## record initial condition output value in timeseries
    p.h_timeseries[0] = p.calc_timeseries_val(p.h_initial)
    ## time loop starting at t_ = p.Dt and j = 1
    for j, t_ in enumerate(p.t[1:], start=1):
        ## explicit update data for u (a)
        S = calc_S(u, h, q, p.Dt, p.Dx, p.theta, p.g)
        ## LHS and RHS for free surface system (b)
        RHS_h = calc_RHS_freesurface(u, S, h, p.Dt, p.Dx, p.Dz, p.theta)
        ## solve for free surface (c)
        h = scipy.sparse.linalg.spsolve(LHS_h, RHS_h)
        ## hydrostatic: q_c = 0
        if p.hydrostatic:
            u = predict_u_velocity(h, S, p.Dt, p.Dx, p.theta, p.g)
            w = update_w_velocity(u, p.Dx, p.Dz)
        ## non hydrostatic q
        else:
            ## update predictor velocity fields using old q (d)
            u_str = predict_u_velocity(h, S, p.Dt, p.Dx, p.theta, p.g)
            w_str = predict_w_velocity(w, q, p.Dt, p.Dz)
            ## LHS, RHS for pressure field
            RHS_q = calc_RHS_pressure(u_str, w_str, p.Dt, p.Dx, p.Dz)
            ## use CG to solve for q_c and update nonhydrostatic pressure
            q_c, info = scipy.sparse.linalg.cg(LHS_q, RHS_q, tol=cg_tol)
            if info != 0:
                warnings.warn("CG solver: info = {}".format(info))
            q_c = q_c.reshape((p.N_i, p.N_k))
            ## add the non-hydrostatic pressure correction
            q += q_c
            ## correct predicted u and w velocity fields
            u = update_u_velocity(u_str, q_c, p.Dt, p.Dx)
            w = update_w_velocity(u, p.Dx, p.Dz)
        ## store timeseries data
        p.h_timeseries[j] = p.calc_timeseries_val(h)
    ## output data at final timestep
    p.h_out = h 
    p.u_out = u 
    p.w_out = w 

def snapshot_velocity_freesurface(x, z, u, w, h, L, D):
    fig = plt.figure()
    X, Z = np.meshgrid(x/L, z/D)
    uc = (u[1:,:] + u[:-1,:]) / 2
    wc = (w[:,1:] + w[:,:-1]) / 2
    plt.plot(x, h + D)
    plt.pcolor(X, Z, uc.T**2 + wc.T**2, cmap='viridis')
    plt.quiver(X, Z, uc.T, wc.T)
    plt.xlabel("x/L")
    plt.ylabel("z/D")
    return fig 

def timeseries_freesurface(t, T0, h_list, a, labels, subplots):
    fig = plt.figure()
    n_sp = max(subplots)
    plt.subplot(n_sp, 1, 1)
    t_ = t / T0
    for j, h in enumerate(h_list):
        plt.subplot(n_sp, 1, subplots[j])
        plt.plot(t_, h/a, label=labels[j])
    plt.xlabel('t/T0')
    plt.ylabel('h/a')
    plt.legend(loc="upper left")
    return fig 

def snapshot_velocity_profiles(z, D, u_list, w_list, labels, subplots):
    pass 