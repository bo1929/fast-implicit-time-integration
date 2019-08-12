from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse.linalg import gmres, eigs
from scipy.sparse import csr_matrix
from numpy.linalg import inv
def cranksolver(boundary):
	###################
	#instance parameters
	c=10# characteristic wave speed
	l = 1                         
	nx =2**8#number of space intervals x-axis                  
	T = 0.1 #time domain                  
	nt = 2**8#number of time intervals
	v_0=0 #inital t=0 v
	std_=0.05
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
	lamb=(c*k)/(4*h)
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
	coef_matrix1=np.zeros((2*(nx-1), 2*(nx-1)))
	coef_matrix2=np.zeros((2*(nx-1), 2*(nx-1)))
	for i in range(nx-2):
		coef_matrix1[i][i]=1
		coef_matrix1[i][i+nx]=-lamb
		coef_matrix1[i+1][i+nx-1]=lamb
	coef_matrix1[nx-2][nx-2]=1
	coef_matrix1[2*nx-3][2*nx-3]=1
	
	for i in range(nx-2,2*nx-3):
		coef_matrix1[i][i]=1
		coef_matrix1[i][i-nx+3]=-lamb
		coef_matrix1[i+1][i-nx+2]=lamb
	
	for i in range(nx-2):
		coef_matrix2[i][i]=1
		coef_matrix2[i][i+nx]=+lamb
		coef_matrix2[i+1][i+nx-1]=-lamb
	coef_matrix2[nx-2][nx-2]=1
	coef_matrix2[2*nx-3][2*nx-3]=1
	
	for i in range(nx-2,2*nx-3):
		coef_matrix2[i][i]=1
		coef_matrix2[i][i-nx+3]=lamb
		coef_matrix2[i+1][i-nx+2]=-lamb
	
	uv_ol = np.append(U_old[1:-1],v_old[1:-1])
	uv_curr =np.append(U_curr[1:-1],v_curr[1:-1])
	kterm=np.zeros(uv_ol.size)

	v0_o=v_init[0]
	vnx_o=v_init[nx]
	u0_o=U_init[0]
	unx_o=U_init[nx]

	v0_c=0
	vnx_c=0
	u0_c=0
	unx_c=0
	inv_coef1=inv(coef_matrix1)
	vecs =eigs(coef_matrix1)
	print(vecs)
	for n in range(1, nt+2):

		v0_c=lamb*4*(uv_ol[nx-1]-v0_o)+v0_o
		vnx_c=lamb*4*(uv_ol[2*nx-3]-vnx_o)+vnx_o
		u0_c=lamb*4*(uv_ol[0]-u0_o)+u0_o
		unx_c=lamb*4*(uv_ol[nx-2]-unx_o)+unx_o
		
		kterm[0]=lamb*(-v0_c-v0_o)
		kterm[nx-2]=lamb*(vnx_o+vnx_c)
		kterm[nx-1]=lamb*(-u0_c-u0_o)
		kterm[2*nx-3]=lamb*(unx_o+unx_c)

		A=csr_matrix(coef_matrix1)
		uv_curr=np.matmul(inv_coef1,(np.matmul(coef_matrix2,uv_ol)+kterm))
		#uv_curr=np.linalg.solve(coef_matrix1,np.matmul(coef_matrix2,uv_ol)+kterm)


		U_curr[0]=u0_c
		U_curr[nx]=unx_c
		U_curr[1:-1]=uv_curr[0:nx-1]
		v_curr[1:-1]=uv_curr[nx-1:2*nx-2]
		
		uv_ol=uv_curr
		U_old=U_curr
		
		v0_c=v0_o
		vnx_c=vnx_o
		u0_c=u0_o
		unx_c=unx_o

		plt.clf()  # Clear the figure
		plt.plot(x, U_curr, color='xkcd:black', label='Current wave')
		plt.plot(x, U_init, color='xkcd:red', label='Initial wave')  # Contour plotting the values
		plt.xlabel('x-axis: space')  # X axis label
		tit='1D wave/ '+ boundary
		plt.title(tit)
		plt.legend()
		plt.grid(True)  # Enabling gridding
		plt.pause(0.001)


cranksolver("open")