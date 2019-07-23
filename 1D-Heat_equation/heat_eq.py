# 1D Heat equation approximation with Backward Difference Scheme

import numpy as np
import matplotlib.pyplot as plt

L = 10                          
nx = 100                        
T = 500                        
nt = 500                        
alpha =10**(-1)                   
x = np.linspace(0, L, nx+1)
u = np.zeros(x.size)
l = np.zeros(x.size)
z = np.zeros(x.size)    
h =L/nx            
t = np.linspace(0, T, nt+1)     
k = T/nt
lambd = alpha*k/(h**2)          
Temp_init = 100*np.sin(np.pi*x/L)
Temp_old = np.copy(Temp_init)
plt.ion()
plt.figure()
l[1]=1+2*lambd
u[1]=(-lambd)/l[1]
for i in range(2,nx-1):
    l[i]=1+2*lambd+lambd*u[i-1]
    u[i]=(-lambd)/l[i]
l[nx-1]=1+2*lambd+lambd*u[nx-2]
for n in range(1, nt):
    print('time t=',t[n])
    z[1]= Temp_old[1]/l[1]
    for i in range(2, nx):
        z[i]=(Temp_old[i]+lambd*z[i-1])/l[i]
    Temp_old[nx-1]=z[nx-1]
    for i in range(1,nx-1):
        Temp_old[nx-1-i]=z[nx-1-i]-u[nx-1-i]*Temp_old[nx-i]
    

    plt.clf()  # Clear the figure
    plt.plot(x, Temp_init, color='xkcd:black', label='Initial Temperature')
    plt.plot(x, Temp_old, color='xkcd:salmon', label='Current Temperature')  # Contour plotting the values
    plt.xlabel('Length', fontsize=10)  # X axis label
    plt.ylabel('Temperature in $^o C$', fontsize=10)  # Y axis label
    plt.title('Temperature Diffusion in a 1D rod')
    # string = " ".join(['Time $t$ = ', str(t[n]), ' s, $T_{max}$ = ', str(np.amax(Temp_old)), ' $^o C$, $T_{min}$ = ', str(np.amin(Temp_old)), '$^o C$'])
    string = 'Time $t$ = '+str(t[n])+' s, $T_{max}$ = '+str(np.amax(Temp_old))+' $^o C$, $T_{min}$ = '+str(np.amin(Temp_old))+'$^o C$'
    plt.suptitle(string)
    plt.legend()
    plt.grid(True)  # Enabling gridding
    plt.axis((-0.1, 10.1, -105, 105))  # Making axis rigid
    plt.pause(0.01)