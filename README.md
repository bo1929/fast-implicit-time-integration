# Fast Implicit Time Integration for 1D PDEs

Small numerical solvers for the 1D heat and wave equations, with an emphasis on implicit time integration and a communication-avoiding Crank-Nicolson variant for the wave equation.

The current codebase is a cleaned-up refactor of the original 2019 project. It is intended to be easy to run locally, inspect, and extend.

## What Is Here

- `solvers/`
  Core implementations for the heat equation, wave equation, time-stepping schemes, and Fourier differentiation.
- `examples/`
  Runnable scripts for each solver and for the communication-avoiding comparison.
- `analysis/`
  A convergence-data generator and plotter for the wave equation.
- `tests/`
  Small regression tests covering the main implicit solver paths.

## Installation

```bash
git clone <repository-url>
cd fast-implicit_time-integration
pip install -r requirements.txt
```

## Quick Start

Run the examples from the repository root:

```bash
python examples/heat_equation.py
python examples/wave_equation.py
python examples/fourier_wave.py
python examples/communication_avoiding.py
```

Run the local regression tests:

```bash
python -m unittest discover -s tests -v
```

Generate convergence data and plots:

```bash
python analysis/convergence.py --generate --c 0.5
python analysis/convergence.py --plot --c 0.5
```

## Wave Equation Model

The wave solver evolves the first-order system

```math
u_t = c\,v_x, \qquad v_t = c\,u_x,
```

which is equivalent to the second-order wave equation

```math
u_{tt} = c^2 u_{xx}.
```

Space is discretized with centered finite differences on a uniform grid:

```math
\partial_x f(x_i) \approx \frac{f_{i+1} - f_{i-1}}{2\Delta x}.
```

For periodic problems, the discrete update can be written in block form as

```math
\mathbf{w}_t = A \mathbf{w},
\qquad
\mathbf{w} =
\begin{bmatrix}
\mathbf{u} \\
\mathbf{v}
\end{bmatrix}.
```

## Time Integrators

The code includes:

- Backward Euler
- Crank-Nicolson
- RK2
- RK4
- An FFT-based analytic/reference propagator for the periodic wave equation

At a high level:

- Backward Euler uses

```math
\left(I - \Delta t\,A\right)\mathbf{w}^{n+1} = \mathbf{w}^n.
```

- Crank-Nicolson uses

```math
\left(I - \frac{\Delta t}{2}A\right)\mathbf{w}^{n+1}
=
\left(I + \frac{\Delta t}{2}A\right)\mathbf{w}^n.
```

Implicit solves are handled with SciPy GMRES.

## Communication-Avoiding Crank-Nicolson

The `comm_cn` method splits the domain into chunks and pads each chunk with halo cells. Each chunk performs a local implicit solve using the overlap data, which reduces how often neighboring chunks must exchange boundary information in a parallel setting.

Parameters:

- `ncx`: chunk size
- `hx`: halo width

Typical usage:

```python
from solvers import Wave1DSolver

solver = Wave1DSolver(c=0.5, L=1.0, T=1.0, nx=256, nt=256)
u, v = solver.solve(method="comm_cn", ncx=16, hx=4)
```

Larger halos generally improve agreement with the full Crank-Nicolson solve, but increase local work.

## Heat Equation Model

The heat solver advances

```math
u_t = \alpha u_{xx}
```

with an implicit backward Euler discretization:

```math
\frac{u_i^{n+1} - u_i^n}{\Delta t}
=
\alpha
\frac{u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}}{\Delta x^2}.
```

In 1D this produces a tridiagonal linear system, solved here with a Thomas-algorithm style sweep.

## Notes

- The wave examples assume periodic boundaries.
- The analytic/reference solution is only available for the periodic wave setup.
- The communication-avoiding implementation is best viewed as a research/demo prototype rather than a production parallel solver.

## Author

Ali Osman Berk Sapci  
Summer 2019, Technical University of Munich
