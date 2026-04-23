"""
Convergence analysis tools for PDE solvers.

Generates convergence plots comparing different time integration schemes
against analytic solutions.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from solvers import Wave1DSolver


DATA_DIR = Path(__file__).resolve().parent


class PlotStyles:
    """Manages plot styles for consistent visualization."""

    def __init__(self):
        self.markers = [".", "1", "2", "3", "4", "+", "x", 4, 5, 6, 7, 8, 9, 10, 11]
        self.linestyles = [
            (0, (1, 1)),
            (0, (5, 10)),
            (0, (5, 5)),
            (0, (5, 1)),
            (0, (3, 10, 1, 10)),
            (0, (3, 5, 1, 5)),
            (0, (3, 1, 1, 1)),
            (0, (3, 5, 1, 5, 1, 5)),
            (0, (3, 10, 1, 10, 1, 10)),
        ]
        self.colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        self.reset()

    def reset(self):
        self.c_colors = 0
        self.c_markers = 0
        self.c_linestyles = 0

    def get_next_style(self):
        retval = {
            "color": self.colors[self.c_colors % len(self.colors)],
            "marker": self.markers[self.c_markers % len(self.markers)],
            "linestyle": self.linestyles[self.c_linestyles % len(self.linestyles)],
        }
        self.c_colors += 1
        self.c_markers += 1
        self.c_linestyles += 1
        return retval


def _load_solution_history(path):
    """Load saved solution history, supporting legacy 1D snapshots."""
    data = np.load(path)
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        return data[np.newaxis, :]
    return data


def _trim_width(num_points):
    """Trim a small boundary band to match the historical post-processing."""
    return min(8, max(1, num_points // 8))


def compute_error(filename_prefix, c, min_deg, max_deg=13):
    """
    Compute error for a given method against analytic solution.

    Loads saved .npy files with naming convention:
    {method}_deg{deg}_c{c}_Co{courant}.npy
    """
    error_levels = np.full(max_deg - min_deg, np.nan)
    x = np.full(max_deg - min_deg, np.nan)

    for m in range(min_deg, max_deg):
        deg = m
        ref_file = DATA_DIR / f"analytic_deg{deg}_c{c}_Co{c}.npy"
        method_file = DATA_DIR / f"{filename_prefix}{deg}_c{c}_Co{c}.npy"

        try:
            u_ref = _load_solution_history(ref_file)
            u_method = _load_solution_history(method_file)
            steps = min(u_ref.shape[0], u_method.shape[0])
            u_ref = u_ref[:steps]
            u_method = u_method[:steps]
            x[m - min_deg] = 2**m
            trim = _trim_width(u_ref.shape[1])
            error = np.abs(u_method[:, trim:-trim] - u_ref[:, trim:-trim])
            error_levels[m - min_deg] = np.max(error)
        except FileNotFoundError:
            continue

    return error_levels, x


def generate_convergence_data(c=0.5, min_deg=5, max_deg=13):
    """
    Generate convergence data by running simulations at different resolutions.

    This generates .npy files that can be analyzed later.
    """
    print("Generating convergence data...")
    print(f"Target Courant number: {c:.2f}")
    print(f"Resolution range: 2^{min_deg} to 2^{max_deg-1}")

    methods = [
        ("analytic", None),
        ("backward_euler", None),
        ("crank_nicolson", None),
        ("rk2", None),
        ("rk4", None),
    ]

    for method, params in methods:
        print(f"\nRunning {method}...")
        for deg in range(min_deg, max_deg):
            nx = 2**deg
            nt = 2**deg

            solver = Wave1DSolver(
                c=c, L=1.0, T=1.0, nx=nx, nt=nt, initial_condition="gaussian", std=0.05
            )

            if params:
                u, _ = solver.solve(method=method, save_history=True, **params)
            else:
                u, _ = solver.solve(method=method, save_history=True)

            filename = DATA_DIR / f"{method}_deg{deg}_c{c}_Co{c}.npy"
            history = np.asarray([state[0][:-1] for state in solver.get_history()], dtype=float)
            if history.size == 0:
                history = np.asarray([u[:-1]], dtype=float)
            with open(filename, "wb") as f:
                np.save(f, history)
            print(f"  deg={deg}: saved to {filename}")


def plot_convergence(c=0.5, min_deg=5):
    """Generate convergence plot from saved data."""
    # Setup plot style
    rcParams.update({"figure.dpi": 100})
    rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(figsize=(10, 6))
    ps = PlotStyles()

    # Plot each method
    methods_to_plot = [
        ("backward_euler_deg", "Backward Euler", ps.get_next_style()),
        ("rk4_deg", "RK4", ps.get_next_style()),
        ("rk2_deg", "RK2", ps.get_next_style()),
        ("crank_nicolson_deg", "Crank-Nicolson", ps.get_next_style()),
    ]

    for prefix, label, style in methods_to_plot:
        try:
            err, x = compute_error(prefix, c, min_deg)
            valid = np.isfinite(err) & np.isfinite(x)
            if not np.any(valid):
                raise FileNotFoundError(f"No saved data found for {label}")
            dx = 1.0 / x[valid]
            ax.plot(dx, err[valid], linewidth=2, markersize=7, **style, label=label)
        except Exception as e:
            print(f"Could not plot {label}: {e}")

    # Reference slopes
    dx_ref = np.array([2 ** (-m) for m in range(min_deg, 12)])
    ax.plot(dx_ref, 2**6 * dx_ref, "k--", alpha=0.5, label="1st order")
    ax.plot(dx_ref, 2**12 * dx_ref**2, "k:", alpha=0.5, label="2nd order")
    ax.plot(dx_ref, 2**24 * dx_ref**4, "-.", color="gray", alpha=0.5, label="4th order")

    # Labels and formatting
    ax.set_xlabel(r"$\Delta x = \Delta t$")
    ax.set_ylabel("Maximum Global Error")
    ax.set_title(f"Convergence Study: c={c}, Courant={c}, T=[0,1], L=[0,1]")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2)
    ax.legend(loc="best")

    plt.tight_layout()
    pdf_path = DATA_DIR / "convergence_plot.pdf"
    png_path = DATA_DIR / "convergence_plot.png"
    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.savefig(png_path, dpi=150)
    print(f"Saved convergence plot to {pdf_path} and {png_path}")
    plt.show()


def main():
    """Main entry point for convergence analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Convergence analysis for PDE solvers")
    parser.add_argument("--generate", action="store_true", help="Generate simulation data")
    parser.add_argument("--plot", action="store_true", help="Generate convergence plot")
    parser.add_argument("--c", type=float, default=0.5, help="Wave speed (default: 0.5)")
    parser.add_argument(
        "--min-deg", type=int, default=5, help="Minimum resolution exponent (default: 5)"
    )

    args = parser.parse_args()

    if args.generate:
        generate_convergence_data(c=args.c, min_deg=args.min_deg)

    if args.plot:
        plot_convergence(c=args.c, min_deg=args.min_deg)

    if not args.generate and not args.plot:
        print("Usage:")
        print("  python convergence.py --generate  # Generate data")
        print("  python convergence.py --plot     # Create plots")
        print("  python convergence.py --generate --plot  # Both")


if __name__ == "__main__":
    main()
