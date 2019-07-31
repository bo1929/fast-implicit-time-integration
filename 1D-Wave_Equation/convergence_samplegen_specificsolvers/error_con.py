#f* means approximation to f
#u*(i+1,j)= (c*dt/2dx)*(v*(i,j+1)-v*(i,j-1))
#v*(i+1,j)= (c*dt/2dx)*(u*(i,j+1)-u*(i,j-1))
# 0<x<l , 0<t<T
#Boundary conditions= u(0,t)= u(l,t)= 0 for 0<t<T
#Initial conditions= u(x,0)= f(x)= sin(pi*x) 0<=x<=l and pd_u(x,0)/pd_t= 0, 0<=x<=l v(x,0)=60 unit distance/ unit time(?)
import numpy as np
import matplotlib.pyplot as plt
import math

c=100
l = 1                         
nx =2**16                      
T = 0.01                 
nt = 2**16                 
v_0=5

#std_=0.1                  

x = np.linspace(0, l, nx+1)
t = np.linspace(0, T, nt+1)

v_init = np.full(x.size,v_0)
v_curr= np.zeros(x.size)
v_old= np.copy(v_init)

h =l/nx
k=T/nt                 

c_=c*k/h
#################
#temp1=np.full(nx+1,np.e)
#temp2=(-0.5)*np.power(((x-(l/2))/std_),2)

#k1=1/(std_*math.sqrt(2*np.pi))
#U_init=k1*np.power(temp1,temp2)
################
U_init= np.sin(2*np.pi*x)
################
U_old=np.copy(U_init)
U_curr=np.zeros(nx+1)

for n in range(1, nt):
	for i in range(1,nx):
		U_curr[nx-i]=U_old[nx-i]+(v_old[nx-i-1]-v_old[nx-i+1])*(-c)*k/(2*h)
		du=np.zeros(nx+1)
	U_curr[nx]=0
	U_curr[0]=0
	
	for i in range(1,nx):
		v_curr[nx-i]=v_old[nx-i]+((U_old[nx-i-1]-U_old[nx-i+1])*(-c)*k)/(2*h)
	v_curr[0]=v_old[0] + c_*(v_old[0+1] - v_old[0]) #open boundary condition
	v_curr[nx]=v_old[nx] - c_*(v_old[nx] - v_old[nx-1]) #open boundary condition
	v_old=v_curr
	U_old=U_curr
	f = open("reference_15.npy", 'ab')
	np.save(f,U_old)
	f.close
print('#################')

