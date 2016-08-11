import numpy as np
from scipy.interpolate import RectBivariateSpline

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
        return vel / np.linalg.norm(vel)

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
                print "exiting"
                break
            # We should come up with a better stopping criteria that puts
            # a point exactly on the zero contour.
        res.append((yl,xl))
    return tuple(res)
