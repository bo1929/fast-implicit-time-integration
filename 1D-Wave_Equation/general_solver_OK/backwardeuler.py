import numpy as np
import matplotlib.pyplot as plt
import math
def solver(boundary):
    ###################
    #instance parameters
    c=100# characteristic wave speed
    l = 1                         
    nx =2**9 #number of separeted space intervals x-axis                  
    T = 0.005#time domain                  
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
    lamb=(c*k)/(2*h) #One dimensional Courant number
    ##################
    #instance: gaussian initial condition for u(0,x)
    temp1=np.full(nx+1,np.e)
    temp2=(-0.5)*np.power(((x-l/2)/std_),2)
    k1=1/(std_*math.sqrt(2*np.pi))
    U_init=k1*np.power(temp1,temp2)
    #################
    #another possible instance
    #U_init= np.sin(2*np.pi*x)
    #################
    
    U_old=np.copy(U_init)#u_(t)
    U_curr=np.zeros(nx+1)#u_(t+1)
    

    coef_matrix=np.zeros((2*(nx-1), 2*(nx-1)))
    

    for i in range(nx-2):
        coef_matrix[i][i]=1
        coef_matrix[i][i+nx]=-lamb
        coef_matrix[i+1][i+nx-1]=lamb
    coef_matrix[nx-2][nx-2]=1
    

    for i in range(nx-1,2*nx-3):
        coef_matrix[i][i]=1
        coef_matrix[i][i-nx+2]=-lamb
        coef_matrix[i+1][i-nx+1]=lamb
    coef_matrix[2*nx-3][2*nx-3]=1
    uv_ol = np.append(U_old[1:-1],v_old[1:-1])
    uandv_rhs =np.append(U_curr[1:-1],v_curr[1:-1])

    if(boundary=="neumann"):
    ####NEUMANN BOUNDARY CONDITION
    	coef_matrix[nx-1][0]=coef_matrix[nx-1][0]+lamb
    	coef_matrix[2*nx-3][nx-2]=coef_matrix[2*nx-3][nx-2]-lamb
    
    if(boundary=="dirichlet"):
        coef_matrix[0][nx-1]=coef_matrix[0][nx-1]+lamb
        coef_matrix[nx-2][2*nx-3]=coef_matrix[nx-2][2*nx-3]-lamb
    
    v0_o=v_init[0]
    vnx_o=v_init[nx]

    u0_o=U_init[0]
    unx_o=U_init[nx]
    v0_c=0
    vnx_c=0
    u0_c=0
    unx_c=0
    for n in range(1, nt+2):#time iteration except t=0
    
        if(boundary=="open"):
        	v0_c=lamb*(uv_ol[nx-1]-v0_o)+v0_o
        	vnx_c=lamb*(uv_ol[2*nx-3]-vnx_o)+vnx_o
        	u0_c=lamb*(uv_ol[0]-u0_o)+u0_o
        	unx_c=lamb*(uv_ol[nx-2]-unx_o)+unx_o

        	uv_ol[0]=uv_ol[0]-lamb*v0_c
        	uv_ol[nx-2]=uv_ol[nx-2]+lamb*vnx_c
        	uv_ol[nx-1]=uv_ol[nx-1]-lamb*u0_c
        	uv_ol[2*nx-3]=uv_ol[2*nx-3]+lamb*unx_c
        	v_curr[0]=v0_c
        	v_curr[nx]=vnx_c
        	U_curr[0]=u0_c
        	U_curr[nx]=unx_c
        
        if(boundary=="periodic"):
        	v0_c=lamb*(uv_ol[nx-1]-v0_o)+v0_o
        	vnx_c=lamb*(uv_ol[2*nx-3]-vnx_o)+vnx_o
        	unx_c=lamb*(uv_ol[nx-2]-unx_o)+unx_o
        	u0_c=unx_c
        	uv_ol[0]=uv_ol[0]-lamb*v0_c
        	uv_ol[nx-2]=uv_ol[nx-2]+lamb*vnx_c
        	uv_ol[nx-1]=uv_ol[nx-1]-lamb*u0_c
        	uv_ol[2*nx-3]=uv_ol[2*nx-3]+lamb*unx_c
        	v_curr[0]=v0_c
        	v_curr[nx]=vnx_c
        	U_curr[nx]=unx_c
        	U_curr[0]=U_curr[nx]
        v0_o=v0_c
        vnx_o=vnx_c
        u0_o=u0_c
        unx_o=unx_c
        if(boundary=="neumann"):
        ####NEUMANN BOUNDARY CONDITION
        	uv_ol[0]=uv_ol[0]-lamb*v_old[0]
        	uv_ol[nx-2]=uv_ol[nx-2]+lamb*v_old[nx]
        	U_curr[0]=U_curr[1]
        	U_curr[nx]=U_curr[nx-1]
        	v_curr[0]=v_old[0]
        	v_curr[nx]=v_old[nx]
        ##############
        if(boundary=="dirichlet"):
        	uv_ol[nx-1]=uv_ol[nx-1]+0 #do nothing u(0,t)=0
        	uv_ol[2*nx-3]=uv_ol[2*nx-3]-0 #do nothing u(l,t)=0
        	U_curr[0]=0
        	U_curr[nx]=0
        	v_curr[0]=v_curr[1]
        	v_curr[nx]=v_curr[nx-1]
       
        uandv_rhs=np.linalg.solve(coef_matrix,uv_ol)
       

        U_curr[1:-1]=uandv_rhs[0:nx-1]
        
        v_curr[1:-1]=uandv_rhs[nx-1:2*nx-2]
        
        #################
        uv_ol=uandv_rhs
        U_old=U_curr
        #################
        plt.clf()  # Clear the figure
        plt.plot(x, U_curr, color='xkcd:black', label='Current wave')
        plt.plot(x, U_init, color='xkcd:red', label='Initial wave')  # Contour plotting the values
        plt.xlabel('x-axis: space')  # X axis label
        tit='1D wave/ '+ boundary
        plt.title(tit)
        plt.legend()
        plt.grid(True)  # Enabling gridding
        plt.pause(0.001)


boundary_condition='neumann'
solver(boundary_condition)

boundary_condition='dirichlet'
#solver(boundary_condition)

boundary_condition="open"
#solver(boundary_condition)


#NOT WORKING
#boundary_condition="periodic"
#solver(boundary_condition)

