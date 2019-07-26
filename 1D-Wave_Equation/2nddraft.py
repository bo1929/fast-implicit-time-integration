#f* means approximation to f
#u*(i+1,j)= (c*dt/2dx)*(v*(i,j+1)-v*(i,j-1))
#v*(i+1,j)= (c*dt/2dx)*(u*(i,j+1)-u*(i,j-1))
# 0<x<l , 0<t<T
#Initial conditions= u(x,0)= f(x)= sin(2*pi*x) 0<=x<=l and pd_u(x,0)/pd_t= 0, 0<=x<=l 
#LOOKS OK , SET DIFFERENT BOUNDARY CONDITIONS, NOT INTERESTING, DON'T KNOW IF IT CAN BE CLASSIFIED, JUST BOTH ZERO??
import numpy as np
import matplotlib.pyplot as plt
import math
c=0.5
l = 1                         
nx =100	                       
T = 1                    
nt = 200                                      
x = np.linspace(0, l, nx+1)
t = np.linspace(0, T, nt+1)
dt = l / nx
dx = T / nt
U_init= np.sin(2*np.pi*x)
U_prev= np.sin(2*np.pi*x)
U_init[x.size-1]=0
U_old= np.copy(U_init)
U_curr=np.zeros(x.size)
for n in range(1, nt+1):
	for i in range (2,U_curr.size-1):
		U_curr[i]=-U_prev[i]+2*U_old[i]+(c*dt/dx)**2*(U_old[i+1]+U_old[i-1]-2*U_old[i])
	U_prev=U_old
	U_old=U_curr
	plt.clf()  # Clear the figure
	plt.plot(x, U_init, color='xkcd:blue', label='Initial Temperature')
	plt.plot(x, U_old, color='xkcd:salmon', label='Current Temperature')  # Contour plotting the values
	plt.xlabel('Length', fontsize=1)  # X axis label
	plt.title('1D wave')
	plt.legend()
	plt.grid(True)  # Enabling gridding
	plt.pause(0.05)
