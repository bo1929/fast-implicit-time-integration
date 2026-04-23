"""
1D Wave Equation Solver with various time integration schemes.
"""

import numpy as np
import math
from .schemes import (
    forward_euler_step,
    backward_euler_step,
    crank_nicolson_step,
    rk2_step,
    rk4_step,
    comm_avoid_crank_nicolson_step,
    L_setup_periodic,
    L_setup_loc,
    setup_k,
    next_analytic,
    solve_linear_system,
)
from scipy.sparse import csr_matrix


class Wave1DSolver:
    """
    Solver for 1D wave equation: u_tt = c^2 * u_xx
    
    Uses first-order system: u_t = v, v_t = c^2 * u_x
    """
    
    def __init__(self, c=0.1, L=1.0, T=10.0, nx=512, nt=512, 
                 initial_condition='gaussian', std=0.05, v0=0.0):
        """
        Initialize wave equation solver.
        
        Parameters:
        -----------
        c : float
            Wave speed
        L : float
            Domain length
        T : float
            Final time
        nx : int
            Number of spatial grid points (power of 2 recommended)
        nt : int
            Number of time steps
        initial_condition : str
            Type of initial condition ('gaussian', 'sine')
        std : float
            Standard deviation for Gaussian initial condition
        v0 : float
            Initial velocity
        """
        self.c = c
        self.L_domain = L
        self.T = T
        self.nx = nx
        self.nt = nt
        self.std = std
        self.v0 = v0
        
        # Setup grid
        self.x = np.linspace(0, L, nx + 1)
        self.t = np.linspace(0, T, nt + 1)
        self.dx = L / nx
        self.dt = T / nt
        self.lamb = c * self.dt / (2 * self.dx)
        
        # Initialize fields
        self.u_init = self._create_initial_condition(initial_condition)
        self.u = np.copy(self.u_init)
        self.v = np.full(self.x.size, v0)
        
        # Store history if needed
        self.history = []
        
    def _create_initial_condition(self, ic_type):
        """Create initial condition."""
        if ic_type == 'gaussian':
            temp1 = np.full(self.x.size, np.e)
            temp2 = (-0.5) * np.power(((self.x - (self.L_domain/2)) / self.std), 2)
            k1 = 1 / (self.std * math.sqrt(2 * np.pi))
            return k1 * np.power(temp1, temp2)
        elif ic_type == 'sine':
            return np.sin(2 * np.pi * self.x)
        else:
            raise ValueError(f"Unknown initial condition: {ic_type}")
    
    def solve(self, method='crank_nicolson', save_history=False, **kwargs):
        """
        Solve the wave equation.
        
        Parameters:
        -----------
        method : str
            Time integration method ('forward_euler', 'backward_euler',
            'crank_nicolson', 'rk2', 'rk4', 'comm_cn')
        save_history : bool
            Whether to save solution at all time steps
        **kwargs : dict
            Additional parameters for specific methods
            
        Returns:
        --------
        u, v : tuple of arrays
            Final solution and velocity
        """
        self.u = np.copy(self.u_init)
        self.v = np.full(self.x.size, self.v0)
        self.history = []
        
        # Setup matrices for implicit methods
        if method in ["backward_euler", "crank_nicolson"]:
            operator_scale = self.lamb if method == "backward_euler" else self.lamb / 2
            L = csr_matrix(L_setup_periodic(self.x.size - 2, operator_scale))
            if method == "crank_nicolson":
                identity = np.identity(L.shape[0])
                L_M = np.asarray(identity - (L.toarray() - identity))
        
        if method == 'comm_cn':
            ncx = kwargs.get('ncx', 4)
            hx = kwargs.get('hx', 2)
            LL = L_setup_loc(ncx, self.lamb/2, hx)
            identity = np.identity(LL.shape[0])
            LLM = identity - (LL - identity)
            LL = csr_matrix(LL)
        
        # Setup for analytic solution
        if method == 'analytic':
            e_ = np.full(self.x.size - 1, np.e)
            k = setup_k(self.u[:-1], self.dx)
            F = self.u[:-1] / 2
            G = self.u[:-1] / 2
            F_f = np.fft.fft(F)
            G_f = np.fft.fft(G)
        
        # Time stepping
        for n in range(1, self.nt + 1):
            if method == 'forward_euler':
                self.u, self.v = forward_euler_step(self.u, self.v, self.lamb)
            elif method == 'backward_euler':
                self.u, self.v = backward_euler_step(self.u, self.v, L)
            elif method == 'crank_nicolson':
                temp = np.concatenate((self.u[1:-1], self.v[1:-1]))
                uv_next = solve_linear_system(L, L_M @ temp)
                self.u[1:-1] = uv_next[: uv_next.size // 2]
                self.v[1:-1] = uv_next[uv_next.size // 2 :]
                self.u[0] = self.u[-2]
                self.u[-1] = self.u[1]
                self.v[0] = self.v[-2]
                self.v[-1] = self.v[1]
            elif method == 'rk2':
                self.u, self.v = rk2_step(self.u, self.v, self.lamb)
            elif method == 'rk4':
                self.u, self.v = rk4_step(self.u, self.v, self.lamb)
            elif method == 'comm_cn':
                ncx = kwargs.get('ncx', 4)
                hx = kwargs.get('hx', 2)
                n_jobs = kwargs.get("n_jobs")
                self.u, self.v = comm_avoid_crank_nicolson_step(
                    self.u, self.v, ncx, hx, LL, LLM, n_jobs=n_jobs
                )
            elif method == 'analytic':
                self.u = next_analytic(F_f, G_f, e_, self.dt, n, k, self.c)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if save_history:
                self.history.append((np.copy(self.u), np.copy(self.v)))
        
        return self.u, self.v
    
    def get_history(self):
        """Return solution history."""
        return self.history
