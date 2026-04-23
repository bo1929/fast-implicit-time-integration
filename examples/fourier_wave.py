"""
Example: 1D Wave Equation with Fourier Pseudospectral Method

Uses FFT for spatial derivatives, achieving spectral accuracy.
Time integration uses leapfrog/central difference scheme.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers.fourier import fourier_derivative_1st
import numpy as np
import matplotlib.pyplot as plt


def solve_fourier_wave(c=0.1, L=1.0, T=1.0, nx=256, nt=256, std=0.05):
    """
    Solve 1D wave equation using Fourier pseudospectral method.

    Parameters:
    -----------
    c : float
        Wave speed
    L : float
        Domain length
    T : float
        Final time
    nx : int
        Number of grid points (power of 2 recommended)
    nt : int
        Number of time steps
    std : float
        Standard deviation for Gaussian initial condition

    Returns:
    --------
    x, u, u_init : arrays
        Spatial grid, final solution, and initial condition
    """
    # Grid setup
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    x = np.arange(0, nx) * dx

    # Initial condition (Gaussian)
    temp1 = np.full(x.size, np.e)
    temp2 = (-0.5) * np.power(((x - (L / 2)) / std), 2)
    k1 = 1 / (std * np.sqrt(2 * np.pi))
    u_init = k1 * np.power(temp1, temp2)

    # Initialize fields for leapfrog
    u_curr = u_init.copy()
    u_old = u_init.copy()
    u_new = np.zeros(nx)

    v_init = np.zeros_like(x)
    v_curr = v_init.copy()
    v_old = v_init.copy()
    v_new = np.zeros(nx)

    # Time stepping with leapfrog
    for it in range(nt):
        # Spatial derivatives via FFT
        ud2p = fourier_derivative_1st(u_curr, dx)
        v_new = v_old + c * dt * 2 * ud2p

        vd2p = fourier_derivative_1st(v_curr, dx)
        u_new = u_old + c * dt * 2 * vd2p

        # Update time levels
        v_old, v_curr = v_curr, v_new
        u_old, u_curr = u_curr, u_new

        # Periodic boundary conditions (handled by FFT)
        u_curr[0] = u_curr[nx - 2]
        u_curr[nx - 1] = u_curr[1]
        v_curr[0] = v_curr[nx - 2]
        v_curr[nx - 1] = v_curr[1]

    return x, u_curr, u_init


def main():
    print("Fourier Pseudospectral Wave Equation Solver")
    print("=" * 50)

    # Solve
    x, u_final, u_init = solve_fourier_wave(c=0.1, L=1.0, T=1.0, nx=256, nt=256, std=0.05)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x[1:], u_init[1:], "k--", label="Initial")
    plt.plot(x[1:], u_final[1:], "b-", label="Final (t=1.0)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("1D Wave Equation: Fourier Pseudospectral Method")
    plt.legend()
    plt.grid(True)
    plt.savefig("fourier_wave.png", dpi=150)
    print("Saved plot to fourier_wave.png")
    plt.show()

    # Resolution study
    print("\nResolution study:")
    resolutions = [64, 128, 256, 512]
    print(f"{'Resolution':>12} | {'Max value':>12}")
    print("-" * 30)
    for nx in resolutions:
        x, u, _ = solve_fourier_wave(c=0.1, L=1.0, T=1.0, nx=nx, nt=nx, std=0.05)
        print(f"{nx:>12} | {np.max(u):>12.6f}")


if __name__ == "__main__":
    main()
