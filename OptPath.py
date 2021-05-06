from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.interpolate import RectBivariateSpline
import math
from scipy.integrate import ode

def optimal_path_2d(travel_time, starting_points, dx, coords,
                 N=100000):
    """Find the optimal paths from the starting_points to the zero contour
    of the array travel_time.

    Solve the equation x_t = - grad t / | grad t |


    Parameters
    ----------
    travel_time : array-like, 2d
        Array-like describing the travel time at each point in the
        domain. Currently only square arrays are supported

    starting_points : sequence of (x,y) starting points
        Sequence of starting points.

    dx : float
       The grid spacing

    coords: array-like, 1d
      Coordinates each point in travel_time. (redundant, this can be
      calculated from dx)

    N: int
       Maximum iterations.

    Returns
    -------
    paths : tuple
        A tuple containing a tuple of points describing the shortest
        path from each starting point.

    """
    grad_t_y, grad_t_x = np.gradient(travel_time, dx)
    if isinstance(travel_time, np.ma.MaskedArray):
        grad_t_y[grad_t_y.mask]=0.0
        grad_t_y = grad_t_y.data
        grad_t_x[grad_t_x.mask]=0.0
        grad_t_x = grad_t_x.data

    gradx_interp = RectBivariateSpline(coords, coords,
                                       grad_t_x)
    grady_interp = RectBivariateSpline(coords, coords,
                                       grad_t_y)

    def get_velocity(position):
        """ return normalized velocity at pos """
        x, y = position
        vel = np.array([gradx_interp(y, x)[0][0],
                        grady_interp(y, x)[0][0]])
        return old_div(vel, np.linalg.norm(vel))

    def euler_point_update(pos, ds):
        return pos - get_velocity(pos) * ds

    def runge_kutta(pos, ds):
        """ Fourth order Runge Kutta point update """
        k1 = ds * get_velocity(pos)
        k2 = ds * get_velocity(pos - k1/2.0)
        k3 = ds * get_velocity(pos - k2/2.0)
        k4 = ds * get_velocity(pos - k3)
        return pos - (k1 + 2*k2 + 2*k3 + k4)/6.0

    res = []
    for starting_point in starting_points:
        x = runge_kutta(starting_point, dx)
        xl, yl = [x[1]], [x[0]]
        for i in range(N):
            xl.append(x[1])
            yl.append(x[0])
            x = runge_kutta(x, dx)
            distance = ((x[1] - xl[-1])**2 + (x[0] - yl[-1])**2)**0.5
            if distance < dx * 0.9999:
                print("exiting")
                break
            # We should come up with a better stopping criteria that puts
            # a point exactly on the zero contour.
        res.append((yl,xl))
    return tuple(res)

# under development: Euler forward implementation of finding optimal path, stops very close to phi=0
def optpath_eulerforward(xs, ys, tt, phi, startpoint):

    # setting up interpolation
    tt_interp=RectBivariateSpline(xs,ys,tt.T)
    phi_interp=RectBivariateSpline(xs,ys,phi.T)

    # initial condition
    currx=startpoint[0]
    curry=startpoint[1]
    signorigphi=np.asscalar(phi_interp.ev(currx,curry)) # current sign of phi so we can detect a change

    # step size
    dsorig=np.asscalar(0.5*(xs[1]-xs[0]))
    ds=dsorig

    # trajectory
    traj=[] # t, x, y
    traj.append((np.asscalar(tt_interp.ev(currx,curry)),currx,curry))

    # while we have not crossed the phi=0 line (and not taken ridiculously small steps) 
    while signorigphi*np.asscalar(phi_interp.ev(currx,curry))>0 and old_div(ds,dsorig)>1e-10:
        gradx=tt_interp.ev(currx,curry,dx=1,dy=0)
        grady=tt_interp.ev(currx,curry,dx=0,dy=1)
        auxgrad=math.sqrt(gradx*gradx+grady*grady) # to normalize
        gradx=old_div(gradx,auxgrad)
        grady=old_div(grady,auxgrad)
        auxcurrx = currx - gradx*ds
        auxcurry = curry - grady*ds
        # if no sign change in phi, go forward
        if signorigphi*np.asscalar(phi_interp.ev(auxcurrx,auxcurry))>0:
            currx=auxcurrx
            curry=auxcurry
            traj.append((np.asscalar(tt_interp.ev(currx,curry)),currx,curry))
        # otherwise abandon and choose smaller step size
        else:
            ds=ds/2.0
    return np.array(traj)

# under development: using SciPy ODE solver to find optimal path, stops at phi=0 (at what precision?)
# inspired a lot by
#    anaconda/lib/python2.7/site-packages/scipy/integrate/tests/test_integrate.py
#    https://stackoverflow.com/questions/24097640/algebraic-constraint-to-terminate-ode-integration-with-scipy
def optpath_scipyode(xs, ys, tt, phi, startpoint):

    # setting up interpolation
    tt_interp=RectBivariateSpline(xs,ys,tt.T)
    phi_interp=RectBivariateSpline(xs,ys,phi.T)

    # initial condition
    t0 = 0.0
    y0 = startpoint 
    signorigphi=np.asscalar(phi_interp.ev(y0[0],y0[1]))

    # trajectory
    ts = []
    ys = []

    # for trajectory and aborting condition
    def solout(t, y):
        # time of the solver is not time in the problem...
        ts.append(tt_interp.ev(y[0],y[1]))
        ys.append(y.copy())
        if not signorigphi*np.asscalar(phi_interp.ev(y[0],y[1]))>0:
            return -1
        else:
            return 0

    # rhs of ODE
    def rhs(t, y):
        gradx=tt_interp.ev(y[0],y[1],dx=1,dy=0)
        grady=tt_interp.ev(y[0],y[1],dx=0,dy=1)
        auxgrad=math.sqrt(gradx*gradx+grady*grady)
        return [old_div(-gradx,auxgrad), old_div(-grady,auxgrad)]

    # the actual integration
    ig = ode(rhs).set_integrator('dopri5')
    ig.set_initial_value(y0, t0)
    ig.set_solout(solout)
    # throws a warning at the moment...
    ret = ig.integrate(1.0e8)
    # what an ugly hack to make a proper np array...
    npts=np.asarray(ts)
    npts.resize((len(ts),1))
    npys=np.asarray(ys)
    return np.concatenate((npts,npys),axis=1)
