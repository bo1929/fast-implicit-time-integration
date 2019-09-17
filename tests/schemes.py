from scipy import linalg
import numpy as np
import math
from scipy.sparse.linalg import gmres, eigs
from scipy.sparse import csr_matrix
###### for analytic solution
def setup_k(f, dx):
    # Length of vector f
    nx = np.size(f)
    kmax = np.pi / dx
    dk = kmax / (nx / 2)
    k = np.arange(float(nx))
    k[: int(nx/2)] = k[: int(nx/2)] * dk 
    k[int(nx/2) :] = k[: int(nx/2)] - kmax
    return k
def next_analytic(F_f,G_f,e_,dt,it,k,c):
    un=np.zeros(e_.size+1)
    un_f=F_f*np.power(e_,(1j*k)*(-1)*c*dt*it)+G_f*np.power(e_,(1j*k)*c*dt*it)
    un[:-1]=np.real(np.fft.ifft(un_f))
    un[-1]=un[1]

    return(un)
######



def L_setup_loc(ncx,lamb,hx):
    cx_hx=ncx+2*hx
    L_size=cx_hx*2
    L=np.zeros((L_size,L_size))
    for i in range(cx_hx-1):
        L[i][i+cx_hx+1]=- lamb
        L[i+1][i+cx_hx]= lamb
        L[i+cx_hx][i+1]=-lamb
        L[i+cx_hx+1][i]=lamb
    L[cx_hx-1][cx_hx]=-lamb
    L[cx_hx][cx_hx-1]=+lamb
    L[L_size-1][0]=-lamb
    L[0][L_size-1]=lamb
    return(L+np.identity(L_size))
def L_setup_periodic(nbnx,lamb):#lamb=c*dt/2dx
    ch_hg=nbnx
    LU_1size=ch_hg*2    
    LU=np.zeros((LU_1size,LU_1size))
    for i in range(ch_hg-1):
        LU[i][i+ch_hg+1]=- lamb
        LU[i+1][i+ch_hg]= lamb
        LU[i+ch_hg][i+1]=-lamb
        LU[i+ch_hg+1][i]=lamb
    LU[ch_hg-1][ch_hg]=-lamb
    LU[ch_hg][ch_hg-1]=+lamb
    LU[LU_1size-1][0]=-lamb
    LU[0][LU_1size-1]=lamb
    return(LU+np.identity(LU_1size))
def cnic(uc,vc,L):
    L=L/2
    un=np.zeros(uc.size)
    vn=np.zeros(vc.size)
    temp=np.append(uc[1:-1],vc[1:-1])
    uv_next=gmres((np.identity(L[0].size)+L),(np.identity(L[0].size)-L)@temp)
    un[1:-1]=uv_next[0][:uv_next[0].size//2]
    vn[1:-1]=uv_next[0][uv_next[0].size//2:]

    un[0]=un[-2]; un[-1]=un[1]
    vn[0]=vn[-2]; vn[-1]=vn[1]
    return(un,vn)
def b_euler(uc,vc,L):
    un=np.zeros(uc.size)
    vn=np.zeros(vc.size)
    identity=np.identity(L[0].size)
    temp=np.append(uc[1:-1],vc[1:-1])
    uv_next=gmres((identity+L),temp)
    un[1:-1]=uv_next[0][:uv_next[0].size//2]
    vn[1:-1]=uv_next[0][uv_next[0].size//2:]
    un[0]=un[-2]; un[-1]=un[1]
    vn[0]=vn[-2]; vn[-1]=vn[1]
    return(un,vn)
def f_euler(uc,vc,lamb):
    un=np.zeros(uc.size)
    vn=np.zeros(vc.size)
    un[1:-1]=(vc[2:]-vc[0:-2])*lamb+uc[1:-1]
    vn[1:-1]=(uc[2:]-uc[0:-2])*lamb+vc[1:-1]
    un[0]=un[-2]; un[-1]=un[1]
    vn[0]=vn[-2]; vn[-1]=vn[1]

    return(un,vn)
def RK4(uc,vc,lamb):
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
    return(ui,vi)
def RK2(uc,vc,lamb):
    ui=np.zeros(uc.size)
    vi=np.zeros(vc.size)
    k1_v=(uc[2:]-uc[:-2])*lamb
    k1_u=(vc[2:]-vc[:-2])*lamb
    ui[1:-1]=uc[1:-1]+(k1_u/2); ui[0]=ui[-2]; ui[-1]=ui[1]
    vi[1:-1]=vc[1:-1]+(k1_v/2); vi[0]=vi[-2]; vi[-1]=vi[1]
    k2_v=(ui[2:]-ui[:-2])*lamb
    k2_u=lamb*(vi[2:]-vi[:-2])
    ui[1:-1]=uc[1:-1]+k2_u; ui[0]=ui[-2]; ui[-1]=ui[1]
    vi[1:-1]=vc[1:-1]+k2_v; vi[0]=vi[-2]; vi[-1]=vi[1]
    return(ui,vi)

def com_avoid_b_euler(uc,vc,ncx,hx,L):
    un=np.zeros(uc.size)
    vn=np.zeros(vc.size)
    #@leftt boundary
    temp1= np.append(np.append(uc[-hx:],uc[:ncx+hx]) , np.append(vc[-hx:],vc[:ncx+hx]))
    temp2=gmres((L+np.identity(L[0].size)),temp1)
    ut=temp2[0][:temp2[0].size//2]
    vt=temp2[0][temp2[0].size//2:]
    un[:ncx]=ut[hx:-hx]
    vn[:ncx]=vt[hx:-hx]
    #@right boundary
    temp1=np.append(np.append(uc[-hx-ncx-1:],uc[:hx-1]),np.append(vc[-hx-ncx-1:],vc[:hx-1]))
    temp2=gmres(L+np.identity(L[0].size),temp1)
    ut=temp2[0][:temp2[0].size//2]
    vt=temp2[0][temp2[0].size//2:]
    un[-ncx-1:-1]=ut[hx:-hx]
    vn[-ncx-1:-1]=vt[hx:-hx]
    for m in range(1,uc.size//ncx-1):
        temp1=np.append(uc[ncx*m-hx:ncx*(m+1)+hx],vc[ncx*m-hx:ncx*(m+1)+hx])
        temp2=gmres(L+np.identity(L[0].size),temp1)
            
        ut=temp2[0][:temp2[0].size//2]
        vt=temp2[0][temp2[0].size//2:]
        un[ncx*m:ncx*(m+1)]=ut[hx:-hx]
        vn[ncx*m:ncx*(m+1)]=vt[hx:-hx]
        un[0]=un[-2]; un[-1]=un[1]
        vn[0]=vn[-2]; vn[-1]=vn[1]
    return(un,vn)




