"""
1D Heat Equation Solver with implicit time integration.
"""

import numpy as np
import matplotlib.pyplot as plt


class Heat1DSolver:
    """
    Solver for 1D heat equation: u_t = alpha * u_xx
    
    Uses Backward Euler with Thomas algorithm (TDMA) for efficient solution
    of the tridiagonal system.
    """
    
    def __init__(self, alpha=0.1, L=10.0, T=500.0, nx=100, nt=500):
        """
        Initialize heat equation solver.
        
        Parameters:
        -----------
        alpha : float
            Thermal diffusivity
        L : float
            Domain length
        T : float
            Final time
        nx : int
            Number of spatial grid intervals
        nt : int
            Number of time steps
        """
        self.alpha = alpha
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt
        
        # Setup grid
        self.x = np.linspace(0, L, nx + 1)
        self.t = np.linspace(0, T, nt + 1)
        self.h = L / nx
        self.k = T / nt
        self.lambd = alpha * self.k / (self.h ** 2)
        
        # Thomas algorithm coefficients
        self._setup_thomas_coefficients()
        
        # Initial condition
        self.u_init = 100 * np.sin(np.pi * self.x / L)
        self.u = np.copy(self.u_init)
        
    def _setup_thomas_coefficients(self):
        """Setup coefficients for Thomas algorithm."""
        self.l = np.zeros(self.x.size)
        self.u_thomas = np.zeros(self.x.size)
        self.z = np.zeros(self.x.size)
        
        # Forward elimination
        self.l[1] = 1 + 2 * self.lambd
        self.u_thomas[1] = (-self.lambd) / self.l[1]
        for i in range(2, self.nx - 1):
            self.l[i] = 1 + 2 * self.lambd + self.lambd * self.u_thomas[i - 1]
            self.u_thomas[i] = (-self.lambd) / self.l[i]
        self.l[self.nx - 1] = 1 + 2 * self.lambd + self.lambd * self.u_thomas[self.nx - 2]
    
    def solve(self, num_steps=None):
        """
        Solve the heat equation.
        
        Parameters:
        -----------
        num_steps : int, optional
            Number of time steps to compute (default: all)
            
        Returns:
        --------
        u : array
            Final temperature distribution
        """
        if num_steps is None:
            num_steps = self.nt
        
        self.u = np.copy(self.u_init)
        
        for n in range(1, min(num_steps + 1, self.nt)):
            # Forward substitution
            self.z[1] = self.u[1] / self.l[1]
            for i in range(2, self.nx):
                self.z[i] = (self.u[i] + self.lambd * self.z[i - 1]) / self.l[i]
            
            # Back substitution
            self.u[self.nx - 1] = self.z[self.nx - 1]
            for i in range(1, self.nx - 1):
                self.u[self.nx - 1 - i] = self.z[self.nx - 1 - i] - \
                                           self.u_thomas[self.nx - 1 - i] * self.u[self.nx - i]
        
        return self.u
    
    def solve_and_animate(self, interval=0.01):
        """
        Solve and animate the solution (interactive).
        
        Parameters:
        -----------
        interval : float
            Pause between frames in seconds
        """
        self.u = np.copy(self.u_init)
        
        plt.ion()
        plt.figure()
        
        for n in range(1, self.nt):
            # Forward substitution
            self.z[1] = self.u[1] / self.l[1]
            for i in range(2, self.nx):
                self.z[i] = (self.u[i] + self.lambd * self.z[i - 1]) / self.l[i]
            
            # Back substitution
            self.u[self.nx - 1] = self.z[self.nx - 1]
            for i in range(1, self.nx - 1):
                self.u[self.nx - 1 - i] = self.z[self.nx - 1 - i] - \
                                           self.u_thomas[self.nx - 1 - i] * self.u[self.nx - i]
            
            # Plot
            plt.clf()
            plt.plot(self.x, self.u_init, color='black', label='Initial Temperature')
            plt.plot(self.x, self.u, color='salmon', label='Current Temperature')
            plt.xlabel('Length', fontsize=10)
            plt.ylabel('Temperature (°C)', fontsize=10)
            plt.title('Temperature Diffusion in a 1D rod')
            plt.suptitle(f'Time t = {self.t[n]:.1f} s, '
                        f'T_max = {np.amax(self.u):.2f} °C, '
                        f'T_min = {np.amin(self.u):.2f} °C')
            plt.legend()
            plt.grid(True)
            plt.axis((-0.1, self.L + 0.1, -105, 105))
            plt.pause(interval)
        
        plt.ioff()
        return self.u
