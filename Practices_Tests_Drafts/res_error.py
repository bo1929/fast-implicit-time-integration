import numpy as np
import math
import matplotlib.pyplot as plot
from matplotlib import animation

#u(x)=sin(x)**2
#u'(x)=4*pi*sin(x)cos(2*pi*x)
# 0 <= x <= 1 u'(1)=u*'(1) and u'(0)=u*'(0) is given

nx=np.arange(1,10,1)
R=np.full(nx.size, 1)
res=nx
dx=R/nx
error=[]
error_a=[]
for j in range(nx.size):
	actual=np.zeros(nx[j]+1)
	actual_d=np.zeros(nx[j]+1)
	diff_app=np.zeros(nx[j]+1)
	approx=np.zeros(nx[j]+1)
	for i in range(nx[j]+1):
		x_i=dx[j]*i
		actual[i]=np.sin(x_i)
		actual_d[i]=np.cos(x_i)
	diff_app[0]=actual_d[0]
	diff_app[nx[j]]=actual_d[nx[j]]
	for i in range(1,nx[j]):
		diff_app[i]=(actual[i+1]-actual[i-1])/(2*dx[j])
	error.append(np.max(np.abs(diff_app-actual_d)))
	approx[0]=actual[0]
	for i in range(1,nx[j]+1):
		approx[i]=approx[i-1]+diff_app[i]*dx[j]
	error_a.append(np.max(np.abs(approx-actual)))


plot.plot(nx[1:],error[1:], color="red")
plot.title('sin(x)**2')
plot.ylabel('Absolute max. errors of approximation of derivative')
plot.xlabel('Resolution')
plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()
plot.plot(nx,error_a, color="red")
plot.title('sin(x)**2')
plot.ylabel('Absolute max. errors of approximation with Euler\'s Method')
plot.xlabel('Resolution')
plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()

#u(x)=e**(x**2)
#u'(x)=2x*e**(x**2)
# 0 <= x <= 3 u'(3)=u*'(3) and u'(0)=u*'(0) is given

nx=np.arange(10000,100000,5000)
R=np.full(nx.size, 3)
res=nx
dx=R/nx
error=[]
error_a=[]
for j in range(nx.size):
	actual=np.zeros(nx[j]+1)
	actual_d=np.zeros(nx[j]+1)
	diff_app=np.zeros(nx[j]+1)
	approx=np.zeros(nx[j]+1)
	for i in range(nx[j]+1):
		x_i=dx[j]*i
		actual[i]=math.exp(x_i**2)
		actual_d[i]=2*x_i*math.exp(x_i**2)
	diff_app[0]=actual_d[0]
	diff_app[nx[j]]=actual_d[nx[j]]
	for i in range(1,nx[j]):
		diff_app[i]=(actual[i+1]-actual[i-1])/(2*dx[j])
	error.append(np.max(np.abs(diff_app-actual_d)))
	approx[0]=actual[0]
	for i in range(1,nx[j]+1):
		approx[i]=approx[i-1]+diff_app[i]*dx[j]
	error_a.append(np.max(np.abs(approx-actual)))

plot.plot(nx,error, color="red")
plot.title('e**(x**2)')
plot.ylabel('Absolute max. errors of approximation of derivative')
plot.xlabel('Resolution')
plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()

plot.plot(nx,error_a, color="red")
plot.title('e**(x**2)')
plot.ylabel('Absolute max. errors of approximation with Euler\'s Method')
plot.xlabel('Resolution')
plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()


