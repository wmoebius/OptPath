import numpy as np
import skfmm
import OptPath as op
import matplotlib.pyplot as plt

# set the problem
X, Y = np.meshgrid(np.linspace(0,100,1001), np.linspace(0,100,1001))

phi=np.ones_like(X)
phi[:,3:10]=-1 # this needs to be robust in order to stop path finding alg. at the moment

speed=np.ones_like(X)
speed[(X-50)**2+(Y-50)**2<36] = 2. 

# solve the FMM part of the problem
tmatrix=skfmm.travel_time(phi,speed,dx=np.asscalar(X[0,1]-X[0,0]))

# now find the optimal path with two methods
nptraj1=op.optpath_eulerforward(np.squeeze(X[0,:]),np.squeeze(Y[:,0]),tmatrix,phi,(60,60))
nptraj2=op.optpath_scipyode(np.squeeze(X[0,:]),np.squeeze(Y[:,0]),tmatrix,phi,(60,60))

# output and compare
plt.close("all")
plt.imshow(speed,extent=[X[0,0],X[0,-1],Y[0,0],Y[-1,0]],origin='lower')
plt.plot(nptraj1[:,1],nptraj1[:,2],'g.')
plt.plot(nptraj2[:,1],nptraj2[:,2],'b-')
plt.show()
