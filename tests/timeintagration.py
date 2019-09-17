from scipy import linalg
import numpy as np
import math
from scipy.sparse.linalg import gmres, eigs
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import schemes as mtd
from joblib import Parallel, delayed
import multiprocessing

#nx must be odd for the communication avoding integration, because chunk size is always chosen even
#v0=0
num_cores = multiprocessing.cpu_count()
def localintegration(m,ncx,hx,uc,vc,LL,LLM):
		temp1=np.append(uc[ncx*m-hx:ncx*(m+1)+hx],vc[ncx*m-hx:ncx*(m+1)+hx])
		temp2=gmres(LL,LLM@temp1)
		ut=temp2[0][:temp2[0].size//2]
		vt=temp2[0][temp2[0].size//2:]
		return(ut[hx:-hx],vt[hx:-hx])

def setup(nx,c,l,T,nt,v0,std,ncx,hx):
	t = np.linspace(0, T, nt) #time grid points, total nt (0,1, ....., nt-1)
	x =np.linspace(0, l, nx)#space grid points, total nt+1 (0,1, ....., nx)
	
	v_init = np.full(x.size,v0)#v_(t_0)

	u_init = np.zeros(x.size)
	temp1=np.full(x.size-1,np.e)
	temp2=(-0.5)*np.power(((x[:-1]-(l/2))/std),2)
	k1=1/(std*math.sqrt(2*np.pi))
	u_init[:-1]=k1*np.power(temp1,temp2)
	u_init[-1]=u_init[1]

	dx=x[1]-x[0]
	dt=t[1]-t[0]
	lamb=c*dt/(2*dx)

	return(x,t,v_init,u_init,lamb,ncx,hx,nt,c,dt,dx)

def integration(config,method):
	dx=config[-1]
	dt=config[-2]
	c=config[-3]
	x=config[0]
	uc=config[3]
	vc=config[2]
	un = np.zeros(x.size)
	vn = np.zeros(x.size)
	lamb=config[4]
	ncx=config[5]
	hx=config[6]
	L=[]
	nt=config[7]
	if(method=="backwardeuler"):
		L1=mtd.L_setup_periodic(x.size-2,lamb)
		L1=csr_matrix(L1)
	if(method=="cranknicholson"):
		L2=mtd.L_setup_periodic(x.size-2,lamb/2)
		identity=np.identity(L2[0].size)	
		L2M=identity-(L2-identity)
		L2=csr_matrix(L2);L2M=csr_matrix(L2M)
	if(method=="comm_cn"):
		LL=mtd.L_setup_loc(ncx,lamb/2,hx)
		identity=np.identity(LL[0].size)
		LLM=identity-(LL-identity)
		LL=csr_matrix(LL)
	if(method=="analytic"):
		e_=np.full(x.size-1,np.e)
		k=mtd.setup_k(config[3][:-1],dx)
		F=config[3][:-1]/2
		G=config[3][:-1]/2
		F_f= np.fft.fft(F)
		G_f= np.fft.fft(G)
	
	for n in range(1, nt+1):
		if(method=="cranknicholson"):
			temp=np.append(uc[1:-1],vc[1:-1])
			uv_next=gmres(L2,L2M@temp)
			un[1:-1]=uv_next[0][:uv_next[0].size//2]
			vn[1:-1]=uv_next[0][uv_next[0].size//2:]
			un[0]=un[-2]; un[-1]=un[1]
			vn[0]=vn[-2]; vn[-1]=vn[1]
		elif(method=="backwardeuler"):
			temp=np.append(uc[1:-1],vc[1:-1])
			uv_next=gmres(L1,temp)
			un[1:-1]=uv_next[0][:uv_next[0].size//2]
			vn[1:-1]=uv_next[0][uv_next[0].size//2:]
			un[0]=un[-2]; un[-1]=un[1]
			vn[0]=vn[-2]; vn[-1]=vn[1]
		elif(method=="forwardeuler"):
			un[1:-1]=(vc[2:]-vc[0:-2])*lamb+uc[1:-1]
			vn[1:-1]=(uc[2:]-uc[0:-2])*lamb+vc[1:-1]
			un[0]=un[-2]; un[-1]=un[1]
			vn[0]=vn[-2]; vn[-1]=vn[1]
		elif(method=="comm_cn"):
			#@left boundary
			temp1= np.append(np.append(uc[-hx:],uc[:ncx+hx]) , np.append(vc[-hx:],vc[:ncx+hx]))
			temp2=gmres((LL),LLM@temp1)
			ut=temp2[0][:temp2[0].size//2]
			vt=temp2[0][temp2[0].size//2:]
			un[:ncx]=ut[hx:-hx]
			vn[:ncx]=vt[hx:-hx]
			#@right boundary
			temp1=np.append(np.append(uc[-hx-ncx-1:],uc[:hx-1]),np.append(vc[-hx-ncx-1:],vc[:hx-1]))
			temp2=gmres(LL,LLM@temp1)
			ut=temp2[0][:temp2[0].size//2]
			vt=temp2[0][temp2[0].size//2:]
			un[-ncx-1:-1]=ut[hx:-hx]
			vn[-ncx-1:-1]=vt[hx:-hx]
			temp=Parallel(n_jobs=num_cores)(delayed(localintegration)(m,ncx,hx,uc,vc,LL,LLM) for m in range(1,uc.size//ncx-1))
			temp=np.asarray(temp)
			un[0]=un[-2]; un[-1]=un[1]
			vn[0]=vn[-2]; vn[-1]=vn[1]
			for m in range(1,uc.size//ncx-1):
				un[ncx*m:ncx*(m+1)]=temp[:,0][m-1]
				vn[ncx*m:ncx*(m+1)]=temp[:,1][m-1]	

		elif(method=="RK2"):
			k1_v=(uc[2:]-uc[:-2])*lamb
			k1_u=(vc[2:]-vc[:-2])*lamb
			un[1:-1]=uc[1:-1]+(k1_u/2); un[0]=un[-2]; un[-1]=un[1]
			vn[1:-1]=vc[1:-1]+(k1_v/2); vn[0]=vn[-2]; vn[-1]=vn[1]
			k2_v=(un[2:]-un[:-2])*lamb
			k2_u=lamb*(vn[2:]-vn[:-2])
			un[1:-1]=uc[1:-1]+k2_u; un[0]=un[-2]; un[-1]=un[1]
			vn[1:-1]=vc[1:-1]+k2_v; vn[0]=vn[-2]; vn[-1]=vn[1]
		elif(method=="RK4"):
			ui=np.zeros(uc.size)
			vi=np.zeros(vc.size)
			k1_v=(uc[2:]-uc[:-2])*lamb
			k1_u=(vc[2:]-vc[:-2])*lamb
			ui[1:-1]=uc[1:-1]+(k1_u/2); ui[0]=ui[-2]; ui[-1]=ui[1]
			vi[1:-1]=vc[1:-1]+(k1_v/2); vi[0]=vi[-2]; vi[-1]=vi[1]
			k2_v=(ui[2:]-ui[:-2])*lamb
			k2_u=lamb*(vi[2:]-vi[:-2])
			ui[1:-1]=uc[1:-1]+(k2_u/2); ui[0]=ui[-2]; ui[-1]=ui[1]
			vi[1:-1]=vc[1:-1]+(k2_v/2); vi[0]=vi[-2]; vi[-1]=vi[1]
			k3_v=(ui[2:]-ui[:-2])*lamb
			k3_u=lamb*(vi[2:]-vi[:-2])
			ui[1:-1]=uc[1:-1]+(k3_u); ui[0]=ui[-2]; ui[-1]=ui[1]
			vi[1:-1]=vc[1:-1]+(k3_v); vi[0]=vi[-2]; vi[-1]=vi[1]
			k4_v=(ui[2:]-ui[:-2])*lamb
			k4_u=lamb*(vi[2:]-vi[:-2])
			ui[1:-1]=uc[1:-1]+(k1_u+2*k2_u+2*k3_u+k4_u)/6; ui[0]=ui[-2]; ui[-1]=ui[1]
			vi[1:-1]=vc[1:-1]+(k1_v+2*k2_v+2*k3_v+k4_v)/6; vi[0]=vi[-2]; vi[-1]=vi[1]
			un=ui
			vn=vi
		elif(method=="analytic"):
			un_f=F_f*np.power(e_,(1j*k)*(-1)*c*dt*n)+G_f*np.power(e_,(1j*k)*c*dt*n)
			un[:-1]=np.real(np.fft.ifft(un_f))
			un[-1]=un[1]
		else:
			print("no such method")

		#######
		uc=un
		vc=vn
		#######
		if(method=="comm_cn"):
			name=method+"_ncx"+str(ncx)+"_hx"+str(hx)+"_deg"+str(deg)+"_c"+str(c)+"_Co"+str(2*lamb)+".npy"
		else:
			name=method+"_deg"+str(deg)+"_c"+str(c)+"_Co"+str(2*lamb)+".npy"


		f = open(name, 'ab')
		np.save(f,un[:-1])
		f.close


for deg in range(12,13):
#############
	nx=2**deg+1
	nt=2**deg+1
	c=0.125
	std=0.05
	l=1
	T=1
	v0=0
	ncx=4
	hx=2
#############
	config=setup(nx,c,l,T,nt,v0,std,ncx,hx)
	#integration(config,"analytic")
	#for i in range(1,4):
	#	for m in range(i):	
	#		hx=2**(m+1)
	#		ncx=2**(i+1)
	#		config=setup(nx,c,l,T,nt,v0,std,ncx,hx)
	#		integration(config,"comm_cn")

	#integration(config,"RK4")
	#integration(config,"RK2")		
	#integration(config,"forwardeuler")
	integration(config,"comm_cn")