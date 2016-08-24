import numpy as np
import pylab as plt

from skfmm import travel_time
from scipy.optimize import fminbound

from OptPath import optimal_path_2d, optpath_scipyode, optpath_eulerforward

# Brachistochrone curve
# see: http://mathworld.wolfram.com/BrachistochroneProblem.html
b=1/np.pi
theta = np.linspace(0,np.pi,300)
x = b*(theta-np.sin(theta))
y = b*(1-np.cos(theta))


N     = 111 # value can be 111 or 1112
assert (np.linspace(0,1.1,N)==1.0).any()
coords, grid_spacing = np.linspace(0, 1.1, N, retstep=True)
X, Y  = np.meshgrid(coords, coords)
phi = np.ones_like(X)
phi[X==1] = 0
phi[X>1] = -1

vel = np.sqrt(Y)
time = travel_time(phi,vel,dx=grid_spacing)
plt.contourf(X,Y,time)
plt.plot(x,y) # plot exact solution as solid line

# starting point for numerical solution
# we cannot start a the origin because the velocity is zero.
y0 = 0.05
theta0 = np.arccos(1-y0/b)
x0 = b*(theta0-np.sin(theta0))
print y0, x0
plt.plot([x0], [y0], "*") # plot starting point

coords = np.linspace(0, 1.1, N)
dx = coords[1]-coords[0]
xp, yp = optimal_path_2d(time, ((x0,y0),), dx, coords)[0]

print "starting forward euler integration"
ft,fx,fy = optpath_eulerforward(np.squeeze(X[0,:]),np.squeeze(Y[:,0]),time,phi,(x0, y0)).T
print "starting scipy integration"
st,sx,sy = optpath_scipyode(np.squeeze(X[0,:]),np.squeeze(Y[:,0]),time,phi,(x0,y0)).T


plt.plot(xp, yp, "o") # plot numerical solution as points.
plt.plot(fx, fy, "o", color="white")
plt.plot(sx, sy, "o", color="black")

plt.gca().set_aspect(1)
plt.axvline(1)
plt.colorbar()

print 2
print 3



# this is one way to find the error but it is problematic near the end
# point in this case.
thetap = np.arccos(1-np.array(yp)/b)
x_exact = b*(thetap-np.sin(thetap))
error = xp-x_exact
print "error metric 1"
print error

def find_theta(x):
    func = lambda theta : abs(x-b*(theta-np.sin(theta)))
    return fminbound(func, 0, np.pi, xtol=1e-12)

thetap2 = np.array(map(find_theta, xp))
# this number should be smaller than the errors calculated
print abs(xp - b*(thetap2-np.sin(thetap2))).max()
y_exact = b*(1-np.cos(thetap2))
print "error metric 2"
print yp-y_exact

print "==== analytic solution to travel time as a function of theta"
# first find the arc-length of the cycloid as a function of theta
# http://www-math.mit.edu/~djk/18_01/chapter18/section02.html
arc_length = 4/np.pi*(1-np.cos(np.pi/2))
dx=x[1:]-x[:-1]
dy=y[1:]-y[:-1]

print "arc-length exact, numerical", arc_length, sum(np.sqrt(dx**2+dy**2))

# next find travel time along this arc-length
# integrate(2*(sin(a/2))/(sqrt((1-cos(a))/pi))/pi,a)
# this is travel time as a function of theta
tt= lambda a : 2*a*np.sin(a/2.0)/np.sqrt(np.pi-np.pi*np.cos(a))
tt0 = tt(np.pi) # total travel time along Brachistochrone curve
# how to use this to check the accuracy of travel_time???

ybar=(y[1:]+y[:-1])/2.0
ntt=sum(np.sqrt(dx**2+dy**2)/np.sqrt(ybar))
print "travel time exact, numerical", tt0, ntt

plt.show()
