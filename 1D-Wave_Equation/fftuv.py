# This is a configuration step for the exercise. Please run it before calculating the derivative!
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
def fourier_derivative_1st(f, dx):
    # Length of vector f
    nx = np.size(f)
    # Initialize k vector up to Nyquist wavenumber 
    kmax = np.pi / dx
    dk = 1 / (nx / 2)
    k = np.arange(float(nx))
    k[: int(nx/2)] = k[: int(nx/2)] * dk 
    k[int(nx/2) :] = k[: int(nx/2)] - kmax
    
    # Fourier derivative
    ff = np.fft.fft(f)
    ff = (1j*k) * ff
    df_num = np.real(np.fft.ifft(ff))
    return df_num

nt    = 2**8    # number of time steps
c     = 0.1 # acoustic velocity [m/s]
L     = 1    # source location
nx    = 2**8   # number of grid points in x 
std_  =0.05
T     =1
v0    =0
# pressure fields Initialization
sp = np.zeros(nx); spnew = sp; spold = sp; sd2p = sp

dx = L/(nx-1)  # calculate space increment
x  = np.arange(0, nx)*dx   # initialize space coordinates
dt = T/(nt-1) # calculate time step from stability criterion


# source time function
# ---------------------------------------------------------------
t   = np.arange(1, nt+1)*dt  # initialize time axis


#spatial source function
# ---------------------------------------------------------------
temp1=np.full(x.size,np.e)
temp2=(-0.5)*np.power(((x-(L/2))/std_),2)
k1=1/(std_*math.sqrt(2*np.pi))
u_init=k1*np.power(temp1,temp2)
u_curr=u_init
u_old=u_init
#####
v_init=temp1=np.full(x.size,v0)
v_curr=v_init
v_old=v_init
for it in range(nt):
    # ----------------------------------------
    # Fourier Pseudospectral Method
    # ----------------------------------------           
    ud2p  = fourier_derivative_1st(u_curr, dx)      # 2nd space derivative     
    v_new = v_old + c* dt*2*ud2p # Time Extrapolation   
    vd2p  = fourier_derivative_1st(v_curr, dx)             
    u_new = u_old + c*dt*2*vd2p
    v_old, v_curr = v_curr, v_new
    u_old, u_curr = u_curr, u_new               # Time levels
    u_curr[0] = u_curr[nx-2]; u_curr[nx-1] = u_curr[1]
    v_curr[0] = v_curr[nx-2]; v_curr[nx-1] = v_curr[1]
    #sp[1] = 0; sp[nx-1] = 0     # set boundaries pressure free

    plt.clf()  # Clear the figure
    plt.plot(x[1:], u_init[1:], color='xkcd:black', label='Initial wave')
    plt.plot(x[1:], u_old[1:], color='xkcd:red', label='Current wave')  # Contour plotting the values
    plt.xlabel('x-axis: space')  # X axis label
    tit='1D wave/ '
    plt.title(tit)
    plt.legend()
    plt.grid(True)  # Enabling gridding
    plt.pause(0.01)