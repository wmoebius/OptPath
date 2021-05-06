from __future__ import print_function
import numpy as np
import pylab as plt

from skfmm import distance

N=1001
x, dx = np.linspace(0, 1, N, retstep=True)
X, Y  = np.meshgrid(x, x)

phi = np.ones_like(X)
phi[Y==0] = 0
mask = np.zeros_like(phi, dtype=bool)
mask[np.logical_and(X>0.5, Y==0.5)] = True
phi = np.ma.MaskedArray(phi,mask)
d = distance(phi,dx=dx)
#plt.contourf(X,Y,d)
#plt.colorbar()
#plt.show()

exact = np.where(np.logical_not(np.logical_and(Y>0.5, X>0.5)),
                 Y,
                 0.5+np.sqrt((X-0.5)**2+(Y-0.5)**2))
exact = np.ma.MaskedArray(exact,mask)

#plt.matshow(d)
#plt.show()
#plt.matshow(exact)
#plt.show()

#plt.matshow(d-exact)
#plt.show()

error = d-exact
error.fill_value = 0
print(abs(error.data).max())
# error seems to be proportional to dx/2 in all cases. I guess this make sense?
