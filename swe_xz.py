"""
solve the 2D-V shallow water equations
"""
import numpy as np 
import scipy.linalg 


class SweXZProblem:
    """
    class to store parameters for swe_xz model run
    """
    def __init__(self, L, D, t_max, N_i, N_k, Dt, hydrostatic=True, h_initial=None):
        """
        set parameter values
        """
        ## domain length
        self.L = L
        ## domain depth
        self.D = D
        ## final time
        self.t_max = t_max 
        ## number of grid cells in horizontal
        self.N_i = N_i
        ## number of grid cells in vertical
        self.N_k = N_k
        ## timestep size
        self.Dt = Dt
        ## x grid cell size
        self.Dx = L / N_i
        ## z grid cell size
        self.Dz = D / N_k
        ## timestep size
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
        ## free surface initial condition
        if h_initial is None:
            ## vector of same length as cell centered x grid
            self.h_initial = np.zeros_like(self.xc)
        else:
            self.h_initial = h_initial

    def set_h_initial(self, h_initial):
        self.h_initial = h_initial

def calc_S():
    """
    S is an N_i+1 by N_k matrix containing explicit data
    for the u velocity field update 
    """
    pass

def calc_RHS_freesurface():
    """
    R is a length N_i vector containing the RHS for the
    free surface height update
    """
    pass

def construct_LHS_freesurface():
    """
    L is a matrix of shape n_c x n_c where n_c = N_i * N_k 
    """
    pass

def predict_u_velocity(q):
    """
    """
    pass

def predict_w_velocity(q):
    """
    """
    pass

def calc_RHS_pressure():
    """
    """
    pass

def construct_LHS_pressure():
    """
    """
    pass

def CG_solve(LHS, RHS):
    """
    """
    pass 

def update_u_velocity(q):
    """
    """
    pass

def update_w_velocity(q):
    """
    """
    pass

def run_swe_xz_problem(p):
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
    A_p = np.zeros_like(u)
    S = np.zeros_like(u) 
    ## time loop starting at t_ = p.Dt
    for t_ in p.t[1:]:
        ## cell centered H
        H_c = h + p.d
        ## explicit update data for u (a)
        S = calc_S()
        ## LHS and RHS for free surface system (b)
        RHS_h = calc_RHS_freesurface()
        LHS_h = construct_LHS_freesurface()
        ## solve for free surface (c)
        h_new = scipy.linalg.solve(LHS_h, RHS_h)
        ## hydrostatic: q = 0
        if p.hydrostatic:
            u = predict_u_velocity(q)
            w = predict_w_velocity(q)
        ## non hydrostatic q
        else:
            ## update predictor velocity fields using old q (d)
            u_str = predict_u_velocity(q)
            w_str = predict_w_velocity(q)
            ## LHS, RHS for pressure field
            RHS_q = calc_RHS_pressure()
            LHS_q = construct_LHS_pressure()
            ## use CG to solve for q_c and update nonhydrostatic pressure
            q_c = CG_solve(LHS_q, RHS_q)
            q += q_c
            ## correct predicted u and w velocity fields
            u = update_u_velocity()
            w = update_w_velocity()






