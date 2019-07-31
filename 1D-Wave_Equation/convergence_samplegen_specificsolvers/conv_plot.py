import numpy as np
import matplotlib.pyplot as plt
import math

ref=14

error_level=[]
x=[]
for m in range(3,14):
	f = open("reference_"+str(ref)+".npy", 'rb')
	po=m
	f1=open("reference_"+str(po)+".npy", 'rb')
	x.append(m)
	glb_er=np.zeros(2**m)
	for i in range(2**m-2):
		temp2=np.zeros(2**m)
		t_ref=np.zeros(2**m)
		t_ins=np.load(f1)
		U_init= np.sin(2*np.pi*np.linspace(0,1,2**po+1))
		plt.clf()  # Clear the figure
		plt.plot(np.linspace(0,1,2**po+1), U_init, color='xkcd:blue', label='Initial wave')
		plt.plot(np.linspace(0,1,2**po+1), t_ins, color='xkcd:red', label='Current wave')  # Contour plotting the values
		plt.xlabel('x-axis', fontsize=1)  # X axis label
		tit='Resolution: 2**'+str(po)
		plt.title(tit)
		plt.legend()
		plt.grid(True)  # Enabling gridding
		plt.pause(0.05)
		for n in range(2**(ref-po)):
			t_ref=np.load(f)
		for k in range(2**m):
			temp2[k]=t_ref[k*2**(ref-po)]
		error=np.abs(t_ins[0:-1]-temp2)
		glb_er[i]=np.max(error)
	error_level.append(np.max(glb_er))
plt.figure()
plt.plot(x, error_level, color='xkcd:blue', label='Initial wave')
plt.xlabel('x-axis', fontsize=2)  # X axis label
plt.title('2**14 reference convergence')
plt.legend()
plt.grid(True)
plt.show()
