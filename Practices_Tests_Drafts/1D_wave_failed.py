import numpy as np
import math
import matplotlib.pyplot as plot
from matplotlib import animation

Rx = 2
nx=500
Rt=100
nt=100
c=4 #coefficient(?)
x=np.linspace(0,Rx,nx+1)
dx=x[1]-x[0]
t=np.linspace(0,Rt,nt+1)
dt=t[1]-t[0]
s=(nt+1,nx+1)
U=np.zeros(s)
for i in range(s[1]):
	U[0][i]=math.sin(math.radians(180*x[i]))
for i in range(s[0]):
	U[i][0]=0.0
	U[i][s[1]-1]=0.0
#plot.plot(x_ax,U[0])
#plot.title('1D wave')
#plot.ylabel('Amplitude = sin(x)')
#plot.grid(True, which='both')
#plot.axhline(y=0, color='k')
#plot.show()
for j in range(1,s[0]):
	for i in range(1,s[1]-1):
		U[j][i]=((U[j-1][i+1]-U[j-1][i-1])*c*dt)/(2*dx)+U[j-1][i]
#for i in range(100):
#	plot.plot(x_ax,U[i])
#	plot.title('1D wave')
#	plot.ylabel('Amplitude = sin(x)')
#	plot.grid(True, which='both')
#	plot.axhline(y=0, color='k')
#	plot.show()

plot.ion()
plot.figure()

for n in range (0,nt+1):
	plot.clf()
	plot.plot(x, U[0],color='xkcd:black',label='Initial wave')
	plot.plot(x, U[0],color='xkcd:salmon',label='Wave at time t')
	plot.xlabel('x-axis')
	plot.ylabel('Amplitude')
	plot.title('1D wave')
	plot.legend()
	plot.grid(True)
	plot.axis((0, 2, -99999, 99999))
	#plot.pause(0.1)