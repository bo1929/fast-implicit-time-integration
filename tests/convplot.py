import numpy as np
import matplotlib.pyplot as plt
import math


plt_show = False

def setup(s=1.0, nrows=1, ncols=1):
	import matplotlib
	matplotlib.rcParams.update({'figure.dpi': 2080})
	matplotlib.rcParams.update({'font.size': 17})

	matplotlib.rcParams.update({'legend.fontsize': 16})
	import __main__ as pc
	pc.default_figsize=(20, 10)
	pc.default_figsize = (pc.default_figsize[0]*s, pc.default_figsize[1]*s)
	import matplotlib.pyplot as plt
	plt.close()
	return plt.subplots(nrows, ncols, figsize=pc.default_figsize)
class PlotStyles:
	def __init__(self):
		self.markers = []
		self.markers += ['.']
		self.markers += ['1', '2', '3', '4', '+', 'x']
		self.markers += [4, 5, 6, 7, 8, 9, 10, 11]
		self.linestyles = [
	(0, (1, 1)),
	(0, (5, 10)),
	(0, (5, 5)),
	(0, (5, 1)),
	(0, (3, 10, 1, 10)),
	(0, (3, 5, 1, 5)),
	(0, (3, 1, 1, 1)),
	(0, (3, 5, 1, 5, 1, 5)),
	(0, (3, 10, 1, 10, 1, 10)),
	(0, (3, 1, 1, 1, 1, 1))
	]
		self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan','tab:black']
		self.reset()
	def reset(self):
		self.c_colors = 0
		self.c_markers = 0
		self.c_linestyles = 0
	def set_color_counter(self, c):
		self.c_colors = c
	def set_marker_counter(self, c):
		self.c_markers = c
	def set_linestyle_counter(self, c):
		self.c_linestyles = c
	def getNextStyle(self):
		retval = {}
		retval['color'] = self.colors[self.c_colors % len(self.colors)]
		retval['marker'] = self.markers[self.c_markers % len(self.markers)]
		retval['linestyle'] = self.linestyles[self.c_linestyles % len(self.linestyles)]
		self.c_colors += 1
		self.c_markers += 1
		self.c_linestyles += 1
		return retval
def error(filename,c,min):
	ref=13
	error_level=np.zeros(13-min)
	x=np.zeros(13-min)
	for m in range(min,13):
		po=m
		f = open("analytic_deg"+str(po)+"_c"+str(c)+"_Co0.5.npy", 'rb')
		f1=open(filename+str(po)+"_c"+str(c)+"_Co0.5.npy", 'rb')
		x[m-min]=2**m
		glb_er=[]
		for i in range(2**(m)):
			t_ins1=np.load(f1)
			t_ref=np.load(f)
			error=np.abs(t_ins1[8:-8]-t_ref[8:-8])
			glb_er.append(np.max(error))
		error_level[m-min]=np.max(glb_er)
	return(error_level,x)


xaxis=[]
temp=error("backwardeuler_deg",0.5,5)
xaxis.append(temp[0])
xaxis.append(error("comm_cn_ncx4_hx2_deg",0.5,5)[0])
xaxis.append(error("comm_cn_ncx8_hx2_deg",0.5,5)[0])
xaxis.append(error("comm_cn_ncx8_hx4_deg",0.5,5)[0])
xaxis.append(error("comm_cn_ncx16_hx2_deg",0.5,5)[0])
xaxis.append(error("comm_cn_ncx16_hx4_deg",0.5,5)[0])
xaxis.append(error("comm_cn_ncx16_hx8_deg",0.5,5)[0])
xaxis.append(error("forwardeuler_deg",0.5,5)[0])
xaxis.append(error("cranknicholson_deg",0.5,5)[0])
xaxis.append(error("RK4_deg",0.5,5)[0])

x=temp[1]
fig, ax = setup()
ps = PlotStyles()
ax.plot(1/x[:-1], xaxis[0][:-1],**ps.getNextStyle(), label='backward euler')
ax.plot(1/x[:-1], xaxis[1][:-1],**ps.getNextStyle(), label='com.av. crank-nicholson, c.size=4, h.size=2')
ax.plot(1/x[:-1], xaxis[2][:-1],**ps.getNextStyle(), label='com.av. crank-nicholson, c.size=8, h.size=2')
ax.plot(1/x[:-1], xaxis[3][:-1],**ps.getNextStyle(), label='com.av. crank-nicholson, c.size=8, h.size=4')	
ax.plot(1/x[:-1], xaxis[4][:-1], **ps.getNextStyle(), label='com.av. crank-nicholson, c.size=16, h.size=2')	
ax.plot(1/x[:-1], xaxis[5][:-1], **ps.getNextStyle(), label='com.av. crank-nicholson, c.size=16, h.size=4')
ax.plot(1/x[:-1], xaxis[6][:-1], **ps.getNextStyle(), label='com.av. crank-nicholson, c.size=16, h.size=8')
ax.plot(1/x[:-1], xaxis[7][:-1], **ps.getNextStyle(), label='Explicit Euler')
ax.plot(1/x[:-1], xaxis[8][:-1], **ps.getNextStyle(), label=' crank-nicholson')
ax.plot(1/x[:-1], xaxis[9][:-1], **ps.getNextStyle(), label='RK4')
ax.plot(1/x[:-5],2**6* 1/x[:-5], color='xkcd:black', label='order1')
ax.plot(1/x[:-5], 2**12*1/x[:-5]*1/x[:-5], color='xkcd:yellow', label='order2')

#ax.plot(1/x,2**8* 1/x, **ps.getNextStyle(), label='order1')
#ax.plot(1/x, 2**12*1/x*1/x, **ps.getNextStyle(), label='order2')
ax.grid()
tit='c=0.5(Velocity), w/ Analytic Reference, Courant Number=0.5, \u0394x=\u0394t, T=[0,1] L=[0,1]'
ax.set(xlabel='\u0394x = \u0394t', ylabel='Maximum Global Error',title=tit)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.set_xscale('log',basex=2)
ax.set_yscale('log',basey=2)
ax.legend()
fig.savefig('c1try.pdf', format='svg', dpi=2080)