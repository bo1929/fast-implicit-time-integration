"""
Fourier spectral methods for spatial derivatives.
"""

import numpy as np


def fourier_derivative_1st(f, dx):
    """
    Compute first derivative using Fourier pseudospectral method.
    
    Parameters:
    -----------
    f : array
        Function values on uniform grid
    dx : float
        Grid spacing
        
    Returns:
    --------
    df_num : array
        Numerical derivative of f
    """
    nx = np.size(f)
    kmax = np.pi / dx
    dk = 1 / (nx / 2)
    k = np.arange(float(nx))
    k[:int(nx/2)] = k[:int(nx/2)] * dk
    k[int(nx/2):] = k[:int(nx/2)] - kmax
    
    ff = np.fft.fft(f)
    ff = (1j * k) * ff
    df_num = np.real(np.fft.ifft(ff))
    return df_num


def fourier_derivative_2nd(f, dx):
    """
    Compute second derivative using Fourier pseudospectral method.
    
    Parameters:
    -----------
    f : array
        Function values on uniform grid
    dx : float
        Grid spacing
        
    Returns:
    --------
    d2f_num : array
        Numerical second derivative of f
    """
    nx = np.size(f)
    kmax = np.pi / dx
    dk = kmax / (nx / 2)
    k = np.arange(float(nx))
    k[:int(nx/2)] = k[:int(nx/2)] * dk
    k[int(nx/2):] = k[:int(nx/2)] - kmax
    
    ff = np.fft.fft(f)
    ff = (-k**2) * ff
    d2f_num = np.real(np.fft.ifft(ff))
    return d2f_num
