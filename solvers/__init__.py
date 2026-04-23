"""
PDE Solvers: Fast Implicit Time Integration Methods

A collection of numerical methods for solving 1D partial differential equations
including heat and wave equations with various time integration schemes.
"""

from .schemes import (
    forward_euler_step,
    backward_euler_step,
    crank_nicolson_step,
    rk2_step,
    rk4_step,
    L_setup_periodic,
    L_setup_loc,
    setup_k,
    next_analytic
)

from .wave_solver import Wave1DSolver
from .heat_solver import Heat1DSolver
from .fourier import fourier_derivative_1st

__version__ = "1.0.0"
