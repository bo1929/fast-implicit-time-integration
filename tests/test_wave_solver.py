import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from analysis import convergence
from solvers import Wave1DSolver


class WaveSolverTests(unittest.TestCase):
    def test_crank_nicolson_runs_and_preserves_shape(self):
        solver = Wave1DSolver(c=0.1, L=1.0, T=0.2, nx=64, nt=64, initial_condition="gaussian")
        u, v = solver.solve(method="crank_nicolson")

        self.assertEqual(u.shape, solver.x.shape)
        self.assertEqual(v.shape, solver.x.shape)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(v)))

    def test_implicit_methods_remain_close_to_analytic_solution(self):
        analytic_solver = Wave1DSolver(
            c=0.1, L=1.0, T=0.2, nx=64, nt=64, initial_condition="gaussian"
        )
        u_exact, _ = analytic_solver.solve(method="analytic")

        for method in ("backward_euler", "crank_nicolson"):
            with self.subTest(method=method):
                solver = Wave1DSolver(
                    c=0.1, L=1.0, T=0.2, nx=64, nt=64, initial_condition="gaussian"
                )
                u_num, _ = solver.solve(method=method)
                error = np.linalg.norm(u_num[1:-1] - u_exact[1:-1]) / np.sqrt(solver.nx)
                self.assertLess(error, 1.0)

    def test_comm_avoiding_tracks_standard_crank_nicolson(self):
        kwargs = dict(c=0.2, L=1.0, T=0.1, nx=64, nt=64, initial_condition="gaussian")
        standard_solver = Wave1DSolver(**kwargs)
        u_standard, _ = standard_solver.solve(method="crank_nicolson")

        comm_solver = Wave1DSolver(**kwargs)
        u_comm, _ = comm_solver.solve(method="comm_cn", ncx=8, hx=2)

        diff = np.linalg.norm(u_comm[1:-1] - u_standard[1:-1]) / np.sqrt(comm_solver.nx)
        self.assertLess(diff, 0.5)


class ConvergenceTests(unittest.TestCase):
    def test_compute_error_reads_saved_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            original_dir = convergence.DATA_DIR
            convergence.DATA_DIR = data_dir
            try:
                ref = np.vstack(
                    [np.linspace(0.0, 1.0, 32), np.linspace(0.5, 1.5, 32)]
                )
                method = ref + 0.125
                np.save(data_dir / "analytic_deg5_c0.5_Co0.5.npy", ref)
                np.save(data_dir / "rk4_deg5_c0.5_Co0.5.npy", method)

                error, x = convergence.compute_error("rk4_deg", c=0.5, min_deg=5, max_deg=6)

                self.assertEqual(x[0], 32)
                self.assertAlmostEqual(error[0], 0.125)
            finally:
                convergence.DATA_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
