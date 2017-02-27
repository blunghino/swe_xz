"""
solve the 2D-V shallow water equations
"""
import numpy as np 


class SweXZProblem:
    """
    class to store parameters for swe_xz model run
    """
    def __init__(self, L, D, t_max, N_i, N_k, N_t, h_initial=None):
        """
        set default parameters
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
        ## number of timesteps
        self.N_t = N_t
        ## x grid cell size
        self.Dx = L / N_i
        ## z grid cell size
        self.Dz = D / N_k
        ## timestep size
        self.Dt = t_max / N_t
        ## cell centered grid x, z
        self.xc = np.arrange(self.Dx/2, L + self.Dx/2, self.Dx)
        self.zc = np.arrange(self.Dz/2, D + self.Dz/2, self.Dz)
        ## meshed cell centered grid
        self.Xc, self.Zc = np.meshgrid(self.xc, self.zc)
        ## timesteps
        self.t = np.arrange(0, t_max+self.Dt, self.Dt)
        ## free surface initial condition
        if h_initial is None:
            self.h_initial = np.zeros_like(self.xc)
        else:
            self.h_initial = h_initial



def run_swe_xz_problem(p):
    """
    steps to compute solution to `p`, a SweXZProblem
    """
    u = np.zeros(p.N_i+1)
    w = np.zeros(p.N_k+1)
    h = p.h_initial.copy()
