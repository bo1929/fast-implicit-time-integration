import numpy as np
import matplotlib.pyplot as plt
import math

def solver(boundary):
	###################
	#instance parameters
	c=100# characteristic wave speed
	l = 1                         
	nx =2**9 #number of separeted space intervals x-axis                  
	T = 00.01 #time domain                  
	nt = 2**9#number of seperated time intervals               
	v_0=0 #inital t=0 v, not necessarily constant(?), It does not affect
	std_=0.1
	###################
	x = np.linspace(0, l, nx+1) #x axis space grid points, total nx+1 (0, 1, ....., nx)
	t = np.linspace(0, T, nt+1) #time grid points, total nt+1 (0,1, ....., nt)
	###################
	v_init = np.full(x.size,v_0)#v_(t0)
	#v_init=x*I_v
	v_curr= np.zeros(x.size)#v_(t+1)
	v_old= np.copy(v_init) #v_(t)
	###################
	h =l/nx #space step size dx
	k=T/nt  #time step size dt
	c_=c*k/h #One dimensional Courant number
	##################
	#instance: gaussian initial condition for u(0,x)
	temp1=np.full(nx+1,np.e)
	temp2=(-0.5)*np.power(((x-(l/2))/std_),2)
	k1=1/(std_*math.sqrt(2*np.pi))
	U_init=k1*np.power(temp1,temp2)
	#################
	#another possible instance
	#U_init= np.sin(2*np.pi*x)
	#################
	U_old=np.copy(U_init)#u_(t)
	U_curr=np.zeros(nx+1)#u_(t+1)
	for n in range(1, nt+2):#time iteration except t=0

		for i in range(1,nx):#space iteration for u except boundary conditions, at t_n
			U_curr[nx-i]=U_old[nx-i]+(v_old[nx-i-1]-v_old[nx-i+1])*(c)*k/(2*h)
		if(boundary=="open"):
			U_curr[0]=U_old[0] + c_*(U_old[0+1] - U_old[0])
			U_curr[nx]=U_old[nx] - c_*(U_old[nx] - U_old[nx-1])
		if(boundary=="drichlect"):
			U_curr[0]=0
			U_curr[nx]=0
		if(boundary=="neumann"):
			U_curr[0]=U_curr[1]
			U_curr[nx]=U_curr[nx-1]
		if(boundary=="periodic"):
				U_curr[nx]=U_old[nx] - c_*(U_old[nx] - U_old[nx-1])
				U_curr[0]=U_curr[nx]
		for i in range(1,nx):#space iteration for v except boundary conditions, at t_n
			v_curr[nx-i]=v_old[nx-i]+((U_old[nx-i-1]-U_old[nx-i+1])*(c)*k)/(2*h)
		if(boundary=="neumann"):
			v_curr[0]=v_old[0]
			v_curr[nx]=v_old[nx]
		if(boundary=="open"):
			v_curr[0]=v_old[0] + c_*(v_old[0+1] - v_old[0])
			v_curr[nx]=v_old[nx] - c_*(v_old[nx] - v_old[nx-1])
		if(boundary=="drichlect"):
			v_curr[0]=v_curr[1]
			v_curr[nx]=v_curr[nx-1]
		if(boundary=="periodic"):
			v_curr[nx]=v_old[nx] - c_*(v_old[nx] - v_old[nx-1])
			v_curr[0]=v_curr[nx]
		
		
		###boundary condition (?): open

		#################
		v_old=v_curr
		U_old=U_curr
		#################
		plt.clf()  # Clear the figure
		plt.plot(x, U_init, color='xkcd:black', label='Initial wave')
		plt.plot(x, U_old, color='xkcd:red', label='Current wave')  # Contour plotting the values
		plt.xlabel('x-axis: space')  # X axis label
		tit='1D wave/ '+ boundary
		plt.title(tit)
		plt.legend()
		plt.grid(True)  # Enabling gridding
		plt.pause(0.001)


boundary_condition='drichlect'
#solver(boundary_condition)

boundary_condition='neumann'
#solver(boundary_condition)

boundary_condition='open'
#solver(boundary_condition)

#periodic works well but mean of gaussian pek should be at boundary
#NOT WORKING
boundary_condition='periodic'
solver(boundary_condition)