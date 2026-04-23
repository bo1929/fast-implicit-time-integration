"""
Example: Communication-Avoiding Crank-Nicolson

Demonstrates domain decomposition approach that reduces communication
overhead in parallel implementations by computing multiple time steps
locally before synchronizing halo regions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers import Wave1DSolver
from solvers.schemes import L_setup_loc
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import time


def run_comm_avoiding_example():
    """
    Run communication-avoiding Crank-Nicolson example.

    Domain is divided into chunks, each solved independently with
    ghost/halo cells for boundary information.
    """
    print("Communication-Avoiding Crank-Nicolson Example")
    print("=" * 50)

    # Parameters
    c = 0.5
    L = 1.0
    T = 1.0
    nx = 256 + 1  # Must be odd for proper chunking
    nt = 256

    # Domain decomposition parameters
    ncx = 16  # Chunk size
    hx = 4  # Halo/ghost cell size (overlap)

    print(f"Domain: [0, {L}] with {nx} points")
    print(f"Chunk size: {ncx}, Halo size: {hx}")
    print(f"Number of chunks: {(nx-1)//ncx}")

    # Create solver
    solver = Wave1DSolver(c=c, L=L, T=T, nx=nx - 1, nt=nt, initial_condition="gaussian", std=0.05)

    # Setup communication-avoiding matrices
    lamb = solver.lamb / 2  # For Crank-Nicolson
    LL = L_setup_loc(ncx, lamb, hx)
    identity = np.identity(LL.shape[0])
    LLM = identity - (LL - identity)
    LL = csr_matrix(LL)

    print(f"\nSolving with comm-avoiding CN...")
    start = time.time()
    u_comm, v_comm = solver.solve(method="comm_cn", ncx=ncx, hx=hx)
    time_comm = time.time() - start
    print(f"Comm-avoiding CN took: {time_comm:.4f}s")

    # Compare with standard Crank-Nicolson
    print(f"\nSolving with standard CN...")
    solver2 = Wave1DSolver(c=c, L=L, T=T, nx=nx - 1, nt=nt, initial_condition="gaussian", std=0.05)
    start = time.time()
    u_std, v_std = solver2.solve(method="crank_nicolson")
    time_std = time.time() - start
    print(f"Standard CN took: {time_std:.4f}s")

    # Compare with analytic
    print(f"\nComputing analytic solution...")
    solver3 = Wave1DSolver(c=c, L=L, T=T, nx=nx - 1, nt=nt, initial_condition="gaussian", std=0.05)
    u_exact, _ = solver3.solve(method="analytic")

    # Compute errors
    error_comm = np.linalg.norm(u_comm[1:-1] - u_exact[1:-1]) / np.sqrt(nx)
    error_std = np.linalg.norm(u_std[1:-1] - u_exact[1:-1]) / np.sqrt(nx)

    print(f"\nL2 Error (comm-avoiding): {error_comm:.6e}")
    print(f"L2 Error (standard CN):   {error_std:.6e}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(solver.x, solver.u_init, "k--", label="Initial")
    axes[0].plot(solver.x, u_comm, "b-", label="Comm-avoiding CN")
    axes[0].set_title("Communication-Avoiding CN")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(solver2.x, solver2.u_init, "k--", label="Initial")
    axes[1].plot(solver2.x, u_std, "r-", label="Standard CN")
    axes[1].set_title("Standard Crank-Nicolson")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(solver.x, np.abs(u_comm - u_exact), "b-", label="Comm-avoiding error")
    axes[2].plot(solver2.x, np.abs(u_std - u_exact), "r--", label="Standard CN error")
    axes[2].set_title("Absolute Error vs Analytic")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("comm_avoiding_comparison.png", dpi=150)
    print("\nSaved comparison plot to comm_avoiding_comparison.png")
    plt.show()

    # Parameter sweep over halo sizes
    print("\n" + "=" * 50)
    print("Parameter sweep: varying halo size")
    halo_sizes = [2, 4, 8, 16]
    errors_hx = []

    for hx_test in halo_sizes:
        solver_test = Wave1DSolver(
            c=c, L=L, T=T, nx=nx - 1, nt=nt, initial_condition="gaussian", std=0.05
        )
        u_test, _ = solver_test.solve(method="comm_cn", ncx=ncx, hx=hx_test)
        err = np.linalg.norm(u_test[1:-1] - u_exact[1:-1]) / np.sqrt(nx)
        errors_hx.append(err)
        print(f"  hx={hx_test:2d}: L2 error = {err:.6e}")


def main():
    run_comm_avoiding_example()


if __name__ == "__main__":
    main()
