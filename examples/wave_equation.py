"""
Example: 1D Wave Equation with various time integration schemes

Solves: u_tt = c^2 * u_xx

Compares different numerical methods:
- Forward Euler (explicit, 1st order)
- Backward Euler (implicit, 1st order)
- Crank-Nicolson (implicit, 2nd order)
- RK2 (explicit, 2nd order)
- RK4 (explicit, 4th order)
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers import Wave1DSolver
import matplotlib.pyplot as plt
import numpy as np


def compare_methods():
    """Compare different time integration schemes."""

    methods = {
        "Forward Euler": "forward_euler",
        "Backward Euler": "backward_euler",
        "Crank-Nicolson": "crank_nicolson",
        "RK2": "rk2",
        "RK4": "rk4",
    }

    # Common parameters
    c = 0.1
    L = 1.0
    T = 1.0
    nx = 128
    nt = 128

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, method) in enumerate(methods.items()):
        solver = Wave1DSolver(c=c, L=L, T=T, nx=nx, nt=nt, initial_condition="gaussian", std=0.1)

        u, v = solver.solve(method=method)

        ax = axes[idx]
        ax.plot(solver.x, solver.u_init, "k--", label="Initial")
        ax.plot(solver.x, u, "b-", label=f"Final (t={T})")
        ax.set_title(f"{name}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend()
        ax.grid(True)

    # Analytic solution
    ax = axes[5]
    solver = Wave1DSolver(c=c, L=L, T=T, nx=nx, nt=nt, initial_condition="gaussian", std=0.1)
    u, v = solver.solve(method="analytic")
    ax.plot(solver.x, solver.u_init, "k--", label="Initial")
    ax.plot(solver.x, u, "g-", label=f"Analytic (t={T})")
    ax.set_title("Analytic Solution (FFT-based)")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True)

    plt.suptitle(f"1D Wave Equation: c={c}, nx={nx}, nt={nt}", fontsize=14)
    plt.tight_layout()
    plt.savefig("wave_comparison.png", dpi=150)
    print("Saved comparison to wave_comparison.png")
    plt.show()


def convergence_study():
    """Study convergence of different methods."""
    print("\nRunning convergence study...")

    c = 0.1
    T = 1.0
    resolutions = [32, 64, 128, 256]

    results = {}

    styles = {
        "forward_euler": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "zorder": 2},
        "crank_nicolson": {
            "marker": "s",
            "linestyle": "--",
            "linewidth": 2.5,
            "zorder": 4,
        },
        "rk4": {"marker": "^", "linestyle": "-.", "linewidth": 2.0, "zorder": 3},
    }

    for method in ["forward_euler", "crank_nicolson", "rk4"]:
        errors = []
        for nx in resolutions:
            nt = nx  # Keep dx = dt for fair comparison
            solver = Wave1DSolver(
                c=c, L=1.0, T=T, nx=nx, nt=nt, initial_condition="gaussian", std=0.1
            )
            solver.solve(method=method, save_history=True)
            numerical_history = [state[0] for state in solver.get_history()]

            analytic_solver = Wave1DSolver(
                c=c, L=1.0, T=T, nx=nx, nt=nt, initial_condition="gaussian", std=0.1
            )
            analytic_solver.solve(method="analytic", save_history=True)
            analytic_history = [state[0] for state in analytic_solver.get_history()]

            trim = min(8, max(1, nx // 8))
            history_errors = []
            for u_num, u_exact in zip(numerical_history, analytic_history):
                diff = u_num[trim:-trim] - u_exact[trim:-trim]
                history_errors.append(np.linalg.norm(diff) / np.sqrt(diff.size))
            error = max(history_errors)
            errors.append(error)

        results[method] = errors

    # Plot convergence
    plt.figure(figsize=(10, 6))
    dx_values = [1.0 / nx for nx in resolutions]

    for method, errors in results.items():
        plt.loglog(
            dx_values,
            errors,
            label=method.replace("_", " ").title(),
            **styles[method],
        )

    # Reference slopes
    dx_arr = np.array(dx_values)
    plt.loglog(dx_arr, 10 * dx_arr, "k--", alpha=0.5, label="1st order slope")
    plt.loglog(dx_arr, 100 * dx_arr**2, "k:", alpha=0.5, label="2nd order slope")
    plt.loglog(dx_arr, 1000 * dx_arr**4, "k-.", alpha=0.5, label="4th order slope")

    plt.xlabel(r"Grid spacing $\Delta x$")
    plt.ylabel("L2 Error")
    plt.title("Convergence Study: 1D Wave Equation")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("convergence_study.png", dpi=150)
    print("Saved convergence study to convergence_study.png")
    plt.show()


def main():
    print("=" * 60)
    print("1D Wave Equation Examples")
    print("=" * 60)

    print("\n1. Comparing different time integration methods...")
    compare_methods()

    print("\n2. Running convergence study...")
    convergence_study()


if __name__ == "__main__":
    main()
