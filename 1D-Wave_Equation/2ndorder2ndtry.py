import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from scipy.fftpack import fft, fftshift

L=1
T=3
nt=150 #number of time intervals
nx=100 #number of space intervals

dt=T/nt
dx=L/nx

c=0.5
c_=c*dt/dx #c_ should be smaller than 1
c_2=c_**2

x = np.linspace(0, L, nx+1)


u   = np.zeros(nx+1)    # Solution array at new time level
u_2 = np.zeros(nx+1)    # Solution at 2 time levels back

std_=0.1

temp1=np.full(nx+1,np.e)
temp2=(-0.5)*np.power(((x-(L/2))/std_),2)

k1=1/(std_*math.sqrt(2*np.pi))
ku1=(-1)/((std_**3)*math.sqrt(2*np.pi))

u_1 =k1*np.power(temp1,temp2) #u(x,0)= (x)=e**((-1/2)*(x-(L/2)/sigma)**2), set u_1 to initial condition
du_1=ku1*((L/2)-x)*np.power(temp1,temp2)
u_init= np.copy(u_1)
#plt.plot(x,du_1)
#plt.title(r"Gaussian window ($\sigma$=) " + str(std_))
#plt.ylabel("Amplitude")
#plt.xlabel("Sample")
#plt.show()
#plt.plot(x,u_1)
#plt.title(r"Gaussian window ($\sigma$=) " + str(std_))
#plt.ylabel("Amplitude")
#plt.xlabel("Sample")
#plt.show()

# Special formula for the first step
for i in range(1, nx):
	u[i] = u_1[i] + dt*du_1[i] + 0.5*c_2*(u_1[i-1] - 2*u_1[i] + u_1[i+1])

#Open boundary conditions
i = 0
u[i] = u_1[i] + c_*(u_1[i+1] - u_1[i])

i = nx
u[i] = u[0]

u_2= u_1
u_1= u
for n in range(2,nt-1):
	i = 0
	u[i] = u_1[i] + c_*(u_1[i+1] - u_1[i])
	i = nx
	u[i] = u[0]
	
	for i in range(1,nx):
		u[i] = - u_2[i] + 2*u_1[i] + c_2*(u_1[i-1] - 2*u_1[i] + u_1[i+1])
	plt.clf()  # Clear the figure
	plt.plot(x, u_init, color='xkcd:blue', label='Initial wave')
	plt.plot(x, -1*u, color='xkcd:salmon', label='Current wave')  # Contour plotting the values
	plt.xlabel('x-axis', fontsize=1)  # X axis label
	plt.title('1D wave')
	plt.legend()
	plt.grid(True)  # Enabling gridding
	plt.pause(0.1)
	u_2 =u_1
	u_1= u