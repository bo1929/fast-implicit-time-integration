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
	c=500# characteristic wave speed
	l = 1                         
	nx =2**8#number of space intervals x-axis                  
	T = 0.6 #time domain                  
	nt = 2**5#number of time intervals
	v_0=0 #inital t=0 v
	std_=0.05
	###################
	x = np.linspace(0, l, nx+1) #x axis space grid points, total nx+1 (0, 1, ....., nx)
	t = np.linspace(0, T, nt+1) #time grid points, total nt+1 (0,1, ....., nt)
	###################
	v_init = np.full(x.size-1,v_0)#v_(t0)
	#v_init=x*I_v
	v_curr= np.zeros(x.size-1)#v_(t+1)
	v_old= np.copy(v_init) #v_(t)
	###################
	h =l/nx #space step size dx
	k=T/nt  #time step size dt
	lamb=(c*k)/(4*h)
	##################
	#instance: gaussian initial condition for u(0,x)
	temp1=np.full(nx,np.e)
	temp2=(-0.5)*np.power(((x[:-1]-(l/2))/std_),2)
	k1=1/(std_*math.sqrt(2*np.pi))
	U_init=k1*np.power(temp1,temp2)
	#################
	#another possible instance
	#U_init= np.sin(2*np.pi*x)
	#################
	U_old=np.copy(U_init)#u_(t)
	U_curr=np.zeros(nx)#u_(t+1)
	coef_matrix1=np.zeros((2*(nx-1), 2*(nx-1)))
	coef_matrix2=np.zeros((2*(nx-1), 2*(nx-1)))
	for i in range(nx-2):
		coef_matrix1[i][i]=1
		coef_matrix1[i][i+nx]=-lamb
		coef_matrix1[i+1][i+nx-1]=lamb
	coef_matrix1[nx-2][nx-2]=1
	coef_matrix1[nx-2][2*nx-4]=lamb

	for i in range(nx-1,2*nx-3):
		coef_matrix1[i][i]=1
		coef_matrix1[i][i-nx+2]=-lamb
		coef_matrix1[i+1][i-nx+1]=lamb
	
	coef_matrix1[2*nx-3][2*nx-3]=1
	coef_matrix1[2*nx-3][0]=-lamb
	coef_matrix1[0][2*nx-3]=lamb
	coef_matrix1[nx-2][nx-1]=-lamb
	coef_matrix1[nx-1][nx-2]=lamb
	
	for i in range(nx-2):
		coef_matrix2[i][i]=1
		coef_matrix2[i][i+nx]=+lamb
		coef_matrix2[i+1][i+nx-1]=-lamb
	coef_matrix2[nx-2][nx-2]=1
	coef_matrix2[nx-2][2*nx-4]=-lamb
	for i in range(nx-1,2*nx-3):
		coef_matrix2[i][i]=1
		coef_matrix2[i][i-nx+2]=lamb
		coef_matrix2[i+1][i-nx+1]=-lamb

	coef_matrix2[2*nx-3][2*nx-3]=1
	coef_matrix2[2*nx-3][0]=lamb
	coef_matrix2[0][2*nx-3]=-lamb
	coef_matrix2[nx-2][nx-1]=lamb
	coef_matrix2[nx-1][nx-2]=-lamb
	
	uv_ol = np.append(U_old[1:],v_old[1:])
	uv_curr =np.append(U_curr[1:],v_curr[1:])
	ident=np.identity(2*(nx-1))
	A=csr_matrix(coef_matrix1)
	#print(coef_matrix1)
	#print(coef_matrix2)

	#if(((coef_matrix1-ident).transpose()==((-1)*(coef_matrix1-ident))).all()):
	#	print("yes")
	
	#LU=coef_matrix2-ident
	#LU=coef_matrix1-ident
	#print(LU)
	#print(linalg.eigvals(LU))
	inv_coef2=inv(coef_matrix2)
	A=csr_matrix(np.matmul(inv_coef2,coef_matrix1))
	#A=csr_matrix(coef_matrix1)
	plt.figure()
	plt.pause(4.5)

	for n in range(1, nt+2):
		
		uv_curr=gmres(A,uv_ol)
		uv_curr=uv_curr[0]
		#print(uv_curr)
		#uv_curr=np.matmul(inv_coef1,(np.matmul(coef_matrix2,uv_ol)+kterm))
		#uv_curr=np.linalg.solve(coef_matrix1,np.matmul(coef_matrix2,uv_ol)+kterm)
		U_curr[1:]=uv_curr[0:nx-1]
		v_curr[1:]=uv_curr[nx-1:2*nx-2]
		
		U_curr[0]=U_curr[nx-2]
		v_curr[0]=v_curr[nx-2]
		
		uv_ol=uv_curr
		U_old=U_curr

		plt.clf()  # Clear the figure
		plt.plot(x[:-1], U_curr, color='xkcd:black', label='Current wave')
		plt.plot(x[:-1], U_init, color='xkcd:green', label='Initial wave')  # Contour plotting the values
		plt.xlabel('x-axis: space')  # X axis label
		tit='1D wave/ periodic/ crank_nicholson '
		plt.title(tit)
		plt.legend()
		plt.grid(True)  # Enabling gridding
		plt.pause(0.00001)


cranksolver("open")