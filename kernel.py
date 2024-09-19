import jax.numpy as jnp
import jax
from jax import vmap
from functools import partial

def gaussian_kernel(x, y, scale):
    z = x - y
    return jnp.exp(-jnp.sum(jnp.square(z)) / (2 *(scale ** 2)))

vmapped_kernel = vmap(vmap(gaussian_kernel, in_axes=(None, 0, None)), in_axes=(0, None, None))

@partial(vmap, in_axes= (0,0))
def precompute_Kzz(ind_points_locs, kernel_param):
    Kzz = vmapped_kernel(ind_points_locs, ind_points_locs, kernel_param)
    Kzz += 1e-6*jnp.eye(Kzz.shape[-1])
    return Kzz

@jax.jit
@partial(vmap, in_axes = (None,0,0))
def precompute_Ktz(times, ind_points_locs, kernel_param):
    Ktz = vmapped_kernel(times, ind_points_locs, kernel_param)
    return Ktz