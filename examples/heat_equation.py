"""
Example: 1D Heat Equation with Backward Euler

Solves: u_t = alpha * u_xx

This example demonstrates the implicit Backward Euler scheme with
Thomas algorithm (TDMA) for efficient tridiagonal system solution.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers import Heat1DSolver
import matplotlib.pyplot as plt


def main():
    # Create solver instance
    solver = Heat1DSolver(
        alpha=0.1,  # Thermal diffusivity
        L=10.0,  # Domain length
        T=500.0,  # Final time
        nx=100,  # Spatial resolution
        nt=500,  # Temporal resolution
    )

    print("Solving 1D Heat Equation...")
    print(f"Parameters: alpha={solver.alpha}, L={solver.L}, T={solver.T}")
    print(f"Grid: nx={solver.nx}, nt={solver.nt}")

    # Solve
    u_final = solver.solve()

    # Plot final solution
    plt.figure(figsize=(10, 6))
    plt.plot(solver.x, solver.u_init, "k--", label="Initial $T_0 = 100\sin(\pi x/L)$")
    plt.plot(solver.x, u_final, "r-", linewidth=2, label=f"Final at t={solver.T}")
    plt.xlabel("Position x")
    plt.ylabel("Temperature (°C)")
    plt.title("1D Heat Equation: Backward Euler Solution")
    plt.legend()
    plt.grid(True)
    plt.savefig("heat_solution.png", dpi=150)
    print("Saved plot to heat_solution.png")
    plt.show()

    # Optionally run animation (uncomment to use)
    # print("\nRunning animation...")
    # solver.solve_and_animate(interval=0.01)


if __name__ == "__main__":
    main()
