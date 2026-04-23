"""
Core time integration schemes for 1D wave equation.

Implements various numerical methods including explicit, implicit, and
communication-avoiding schemes.
"""

import numpy as np
from scipy.sparse.linalg import gmres
from scipy.sparse import csr_matrix


def solve_linear_system(matrix, rhs, *, atol=1e-12, rtol=1e-10, maxiter=None):
    """
    Solve a linear system with GMRES and return a flat NumPy array.

    Parameters
    ----------
    matrix : array-like or sparse matrix
        Linear operator.
    rhs : array-like
        Right-hand side vector.
    atol, rtol : float
        GMRES tolerances.
    maxiter : int, optional
        Maximum number of iterations passed to SciPy.
    """
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    solution, info = gmres(matrix, rhs, atol=atol, rtol=rtol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"GMRES failed to converge (info={info})")
    return np.asarray(solution, dtype=float).reshape(-1)


def setup_k(f, dx):
    """
    Setup wavenumber vector for Fourier analysis.
    
    Parameters:
    -----------
    f : array
        Input function values
    dx : float
        Spatial step size
        
    Returns:
    --------
    k : array
        Wavenumber vector
    """
    nx = np.size(f)
    kmax = np.pi / dx
    dk = kmax / (nx / 2)
    k = np.arange(float(nx))
    k[: int(nx/2)] = k[: int(nx/2)] * dk
    k[int(nx/2):] = k[: int(nx/2)] - kmax
    return k


def next_analytic(F_f, G_f, e_, dt, it, k, c):
    """
    Compute next time step using analytic solution via FFT.
    
    Parameters:
    -----------
    F_f, G_f : array
        Fourier coefficients of initial conditions
    e_ : array
        Base array for exponential computation
    dt : float
        Time step
    it : int
        Current time iteration
    k : array
        Wavenumber vector
    c : float
        Wave speed
        
    Returns:
    --------
    un : array
        Solution at next time step
    """
    un = np.zeros(e_.size + 1)
    un_f = F_f * np.power(e_, (1j*k) * (-1) * c * dt * it) + \
           G_f * np.power(e_, (1j*k) * c * dt * it)
    un[:-1] = np.real(np.fft.ifft(un_f))
    un[-1] = un[1]
    return un


def L_setup_periodic(nbnx, lamb):
    """
    Setup system matrix for periodic boundary conditions.
    
    Parameters:
    -----------
    nbnx : int
        Number of interior grid points
    lamb : float
        Courant-like number: c*dt/(2*dx)
        
    Returns:
    --------
    LU : array
        System matrix for implicit schemes
    """
    ch_hg = nbnx
    LU_1size = ch_hg * 2
    LU = np.zeros((LU_1size, LU_1size))
    for i in range(ch_hg - 1):
        LU[i][i + ch_hg + 1] = -lamb
        LU[i + 1][i + ch_hg] = lamb
        LU[i + ch_hg][i + 1] = -lamb
        LU[i + ch_hg + 1][i] = lamb
    LU[ch_hg - 1][ch_hg] = -lamb
    LU[ch_hg][ch_hg - 1] = lamb
    LU[LU_1size - 1][0] = -lamb
    LU[0][LU_1size - 1] = lamb
    return LU + np.identity(LU_1size)


def L_setup_loc(ncx, lamb, hx):
    """
    Setup local system matrix for communication-avoiding scheme.
    
    Parameters:
    -----------
    ncx : int
        Chunk size for domain decomposition
    lamb : float
        Courant-like number
    hx : int
        Halo/ghost cell size
        
    Returns:
    --------
    L : array
        Local system matrix
    """
    cx_hx = ncx + 2 * hx
    L_size = cx_hx * 2
    L = np.zeros((L_size, L_size))
    for i in range(cx_hx - 1):
        L[i][i + cx_hx + 1] = -lamb
        L[i + 1][i + cx_hx] = lamb
        L[i + cx_hx][i + 1] = -lamb
        L[i + cx_hx + 1][i] = lamb
    L[cx_hx - 1][cx_hx] = -lamb
    L[cx_hx][cx_hx - 1] = lamb
    L[L_size - 1][0] = -lamb
    L[0][L_size - 1] = lamb
    return L + np.identity(L_size)


def forward_euler_step(uc, vc, lamb):
    """
    Forward Euler explicit time step.
    
    First-order accurate in time.
    
    Parameters:
    -----------
    uc, vc : array
        Current solution and velocity
    lamb : float
        Courant-like number: c*dt/(2*dx)
        
    Returns:
    --------
    un, vn : tuple of arrays
        Solution at next time step
    """
    un = np.zeros(uc.size)
    vn = np.zeros(vc.size)
    un[1:-1] = (vc[2:] - vc[0:-2]) * lamb + uc[1:-1]
    vn[1:-1] = (uc[2:] - uc[0:-2]) * lamb + vc[1:-1]
    un[0] = un[-2]
    un[-1] = un[1]
    vn[0] = vn[-2]
    vn[-1] = vn[1]
    return un, vn


def backward_euler_step(uc, vc, L):
    """
    Backward Euler implicit time step using GMRES.
    
    First-order accurate in time, unconditionally stable.
    
    Parameters:
    -----------
    uc, vc : array
        Current solution and velocity
    L : array
        System matrix
        
    Returns:
    --------
    un, vn : tuple of arrays
        Solution at next time step
    """
    un = np.zeros(uc.size)
    vn = np.zeros(vc.size)
    temp = np.concatenate((uc[1:-1], vc[1:-1]))
    uv_next = solve_linear_system(L, temp)
    un[1:-1] = uv_next[: uv_next.size // 2]
    vn[1:-1] = uv_next[uv_next.size // 2 :]
    un[0] = un[-2]
    un[-1] = un[1]
    vn[0] = vn[-2]
    vn[-1] = vn[1]
    return un, vn


def crank_nicolson_step(uc, vc, L):
    """
    Crank-Nicolson implicit time step using GMRES.
    
    Second-order accurate in time, unconditionally stable.
    
    Parameters:
    -----------
    uc, vc : array
        Current solution and velocity
    L : array
        System matrix (will be halved internally)
        
    Returns:
    --------
    un, vn : tuple of arrays
        Solution at next time step
    """
    un = np.zeros(uc.size)
    vn = np.zeros(vc.size)
    temp = np.concatenate((uc[1:-1], vc[1:-1]))
    identity = csr_matrix(np.identity(L.shape[0]))
    lhs = identity + L
    rhs = (identity - L) @ temp
    uv_next = solve_linear_system(lhs, rhs)
    un[1:-1] = uv_next[: uv_next.size // 2]
    vn[1:-1] = uv_next[uv_next.size // 2 :]
    un[0] = un[-2]
    un[-1] = un[1]
    vn[0] = vn[-2]
    vn[-1] = vn[1]
    return un, vn


def rk2_step(uc, vc, lamb):
    """
    Runge-Kutta 2nd order (midpoint) time step.
    
    Second-order accurate in time.
    
    Parameters:
    -----------
    uc, vc : array
        Current solution and velocity
    lamb : float
        Courant-like number
        
    Returns:
    --------
    ui, vi : tuple of arrays
        Solution at next time step
    """
    ui = np.zeros(uc.size)
    vi = np.zeros(vc.size)
    
    # Stage 1
    k1_v = (uc[2:] - uc[:-2]) * lamb
    k1_u = (vc[2:] - vc[:-2]) * lamb
    ui[1:-1] = uc[1:-1] + (k1_u / 2)
    ui[0] = ui[-2]
    ui[-1] = ui[1]
    vi[1:-1] = vc[1:-1] + (k1_v / 2)
    vi[0] = vi[-2]
    vi[-1] = vi[1]
    
    # Stage 2
    k2_v = (ui[2:] - ui[:-2]) * lamb
    k2_u = lamb * (vi[2:] - vi[:-2])
    ui[1:-1] = uc[1:-1] + k2_u
    ui[0] = ui[-2]
    ui[-1] = ui[1]
    vi[1:-1] = vc[1:-1] + k2_v
    vi[0] = vi[-2]
    vi[-1] = vi[1]
    
    return ui, vi


def rk4_step(uc, vc, lamb):
    """
    Runge-Kutta 4th order time step.
    
    Fourth-order accurate in time.
    
    Parameters:
    -----------
    uc, vc : array
        Current solution and velocity
    lamb : float
        Courant-like number
        
    Returns:
    --------
    ui, vi : tuple of arrays
        Solution at next time step
    """
    ui = np.zeros(uc.size)
    vi = np.zeros(vc.size)
    
    # Stage 1
    k1_v = (uc[2:] - uc[:-2]) * lamb
    k1_u = (vc[2:] - vc[:-2]) * lamb
    ui[1:-1] = uc[1:-1] + (k1_u / 2)
    ui[0] = ui[-2]
    ui[-1] = ui[1]
    vi[1:-1] = vc[1:-1] + (k1_v / 2)
    vi[0] = vi[-2]
    vi[-1] = vi[1]
    
    # Stage 2
    k2_v = (ui[2:] - ui[:-2]) * lamb
    k2_u = lamb * (vi[2:] - vi[:-2])
    ui[1:-1] = uc[1:-1] + (k2_u / 2)
    ui[0] = ui[-2]
    ui[-1] = ui[1]
    vi[1:-1] = vc[1:-1] + (k2_v / 2)
    vi[0] = vi[-2]
    vi[-1] = vi[1]
    
    # Stage 3
    k3_v = (ui[2:] - ui[:-2]) * lamb
    k3_u = lamb * (vi[2:] - vi[:-2])
    ui[1:-1] = uc[1:-1] + k3_u
    ui[0] = ui[-2]
    ui[-1] = ui[1]
    vi[1:-1] = vc[1:-1] + k3_v
    vi[0] = vi[-2]
    vi[-1] = vi[1]
    
    # Stage 4
    k4_v = (ui[2:] - ui[:-2]) * lamb
    k4_u = lamb * (vi[2:] - vi[:-2])
    ui[1:-1] = uc[1:-1] + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
    ui[0] = ui[-2]
    ui[-1] = ui[1]
    vi[1:-1] = vc[1:-1] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    vi[0] = vi[-2]
    vi[-1] = vi[1]
    
    return ui, vi


def comm_avoid_crank_nicolson_step(uc, vc, ncx, hx, LL, LLM, n_jobs=None):
    """
    Communication-avoiding Crank-Nicolson step with domain decomposition.
    
    Divides domain into chunks and solves locally with halo cells to reduce
    communication in parallel implementations.
    
    Parameters:
    -----------
    uc, vc : array
        Current solution and velocity
    ncx : int
        Chunk size
    hx : int
        Halo/ghost cell size
    LL : sparse matrix
        Local system matrix
    LLM : sparse matrix
        Modified system matrix
        
    Returns:
    --------
    un, vn : tuple of arrays
        Solution at next time step
    """
    from joblib import Parallel, delayed
    import multiprocessing

    num_chunks = max(0, uc.size // ncx - 2)
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), max(1, num_chunks))

    un = np.zeros(uc.size)
    vn = np.zeros(vc.size)
    
    # Left boundary
    temp1 = np.append(np.append(uc[-hx:], uc[:ncx+hx]),
                      np.append(vc[-hx:], vc[:ncx+hx]))
    temp2 = solve_linear_system(LL, LLM @ temp1)
    ut = temp2[: temp2.size // 2]
    vt = temp2[temp2.size // 2 :]
    un[:ncx] = ut[hx:-hx]
    vn[:ncx] = vt[hx:-hx]
    
    # Right boundary
    temp1 = np.append(np.append(uc[-hx-ncx-1:], uc[:hx-1]),
                      np.append(vc[-hx-ncx-1:], vc[:hx-1]))
    temp2 = solve_linear_system(LL, LLM @ temp1)
    ut = temp2[: temp2.size // 2]
    vt = temp2[temp2.size // 2 :]
    un[-ncx-1:-1] = ut[hx:-hx]
    vn[-ncx-1:-1] = vt[hx:-hx]
    
    # Parallel interior solves
    def localintegration(m, ncx, hx, uc, vc, LL, LLM):
        temp1 = np.append(uc[ncx*m-hx:ncx*(m+1)+hx],
                          vc[ncx*m-hx:ncx*(m+1)+hx])
        temp2 = solve_linear_system(LL, LLM @ temp1)
        ut = temp2[: temp2.size // 2]
        vt = temp2[temp2.size // 2 :]
        return ut[hx:-hx], vt[hx:-hx]
    
    if num_chunks > 0:
        if n_jobs == 1:
            temp = [
                localintegration(m, ncx, hx, uc, vc, LL, LLM)
                for m in range(1, uc.size // ncx - 1)
            ]
        else:
            temp = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(localintegration)(m, ncx, hx, uc, vc, LL, LLM)
                for m in range(1, uc.size // ncx - 1)
            )
        temp = np.asarray(temp, dtype=float)
    else:
        temp = np.empty((0, 2, 0))
    
    un[0] = un[-2]
    un[-1] = un[1]
    vn[0] = vn[-2]
    vn[-1] = vn[1]
    
    for m in range(1, uc.size // ncx - 1):
        un[ncx*m:ncx*(m+1)] = temp[m - 1, 0]
        vn[ncx*m:ncx*(m+1)] = temp[m - 1, 1]
    
    return un, vn
