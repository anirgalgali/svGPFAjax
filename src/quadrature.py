import numpy as np
from numpy.polynomial.legendre import leggauss
import jax.numpy as jnp

def leggaussVarLimits(n, a, b):
    """
    Computes n weights and points for Gauss-Legendre numerical integration of a
    function in the interval (a,b).
    """
    x, w = leggauss(deg=n)
    x = np.asarray(x)
    w = np.asarray(w)
    xVarLimits = (x*(b-a)+(b+a))/2
    wVarLimits = (b-a)/2*w
    return xVarLimits, wVarLimits

def getLegQuadPointsAndWeights(n_quad, trials_start_times, trials_end_times):
    n_trials = len(trials_start_times)
    assert(n_trials == len(trials_end_times))
    leg_quad_points = np.empty((n_trials, n_quad), dtype=np.float32)
    leg_quad_weights = np.empty((n_trials, n_quad), dtype=np.float32)

    for r in range(n_trials):
        points, weights = leggaussVarLimits(
                    n=n_quad, a=trials_start_times[r], b=trials_end_times[r])
        leg_quad_points[r,:]  = points
        leg_quad_weights[r,:] = weights
    
    return jnp.asarray(leg_quad_points), jnp.asarray(leg_quad_weights)