#f* means approximation to f
#u*(i+1,j)= (c*dt/2dx)*(v*(i,j+1)-v*(i,j-1))
#v*(i+1,j)= (c*dt/2dx)*(u*(i,j+1)-u*(i,j-1))
# 0<x<l , 0<t<T
#Boundary conditions= u(0,t)= u(l,t)= 0 for 0<t<T
#Initial conditions= u(x,0)= f(x)= sin(2*pi*x) 0<=x<=l and pd_u(x,0)/pd_t= 0, 0<=x<=l v(x,0)=60 unit distance/ unit time(?)
import numpy as np
import matplotlib.pyplot as plt
import math
c=0.5
l = 1                         
nx =100	                       
T = 5                    
nt = 250                  
v_0=5000
std_=0.1                  
x = np.linspace(0, l, nx+1)
#print(x.size)
t = np.linspace(0, T, nt+1)
#print(t.size)
v_init = np.full(x.size,v_0)
#print(v_init)
#v_init = np.sin(np.pi*x)

v_curr= np.zeros(x.size)
v_old= np.copy(v_init)
v_init[0]=0
v_init[x.size-1]=0
h =l/nx
#print(h)
k=T/nt                 
#print(k)
c_=c*k/h
#################
#temp1=np.full(nx+1,np.e)
#temp2=(-0.5)*np.power(((x-(l/2))/std_),2)

#k1=1/(std_*math.sqrt(2*np.pi))
#U_init=k1*np.power(temp1,temp2)
################
U_init= np.sin(2*np.pi*x)

U_old= np.copy(U_init)
U_curr=np.zeros(x.size)
for n in range(1, nt):
	#U_curr[0]=np.sin(2*np.pi*n*k)
	#U_curr[x.size-1]=np.sin(2*np.pi*n*k)
	for i in range(1,nx):
		#print((v_old[i+1]-v_old[i-1])*c*k/(2*h))
		U_curr[i]=U_old[i]+(v_old[i+1]-v_old[i-1])*c*k/(2*h)
	U_curr[0]=U_old[0] + c_*(U_old[0+1] - U_old[0])
	U_curr[nx]=U_old[nx] - c_*(U_old[nx] - U_old[nx-1])
	#U_curr[nx]=U_curr[0]
	U_old=U_curr
	for i in range(1,x.size-1):
		v_curr[i]=v_old[i]+((U_old[i+1]-U_old[i-1])*(c)*k)/(2*h)
	v_curr[0]=v_old[0] + c_*(v_old[0+1] - v_old[0])
	v_curr[nx]=v_old[nx] - c_*(v_old[nx] - v_old[nx-1])
	v_old=v_curr
	plt.clf()  # Clear the figure
	plt.plot(x, U_init, color='xkcd:blue', label='Initial wave')
	plt.plot(x, U_old, color='xkcd:red', label='Current wave')  # Contour plotting the values
	plt.xlabel('x-axis', fontsize=1)  # X axis label
	plt.title('1D wave')
	plt.legend()
	plt.grid(True)  # Enabling gridding
	plt.pause(0.05)
