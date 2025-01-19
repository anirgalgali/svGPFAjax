import jax.numpy as jnp
from jax import vmap
from functools import partial
from kernel import precompute_Ktz
import jax

@jax.jit
@partial(vmap, in_axes=(None,None,1,0), out_axes=1)
def compute_latent_means(kernel_param, ind_points_locs, meanFactor, times):
    
    Ktz = precompute_Ktz(times, ind_points_locs, kernel_param)
    latent_mean = jnp.einsum('ijk,ik -> ij',Ktz, meanFactor)
    # if times.shape[0] == 100:
    # #     jax.debug.print("Ktz inner shape = {}",Ktz.shape)
    # #     jax.debug.print("mean factor inner shape = {}",meanFactor.shape)
    # #     jax.debug.print("latent mean shape = {}",latent_mean.shape)
    #     jax.debug.print("latent mean = {}",latent_mean[:,0])

    return latent_mean

@jax.jit
@partial(vmap, in_axes = (None, None, None, None, 1 ,0),out_axes = 1)
def compute_latent_variances(kernel_param, ind_points_locs, Kzz, Kzz_cho, vCov, times):
    
    Ktz = precompute_Ktz(times, ind_points_locs, kernel_param)
    M = jax.scipy.linalg.cho_solve((Kzz_cho,True), Ktz.transpose(0,2,1))
    # jax.debug.print("vCov inner shape = {}",vCov.shape)
    M_ = jnp.einsum('bik,bkj -> bij',vCov - Kzz, M)
    # jax.debug.print("M_ = {}",M_)  
    # jax.debug.print("M_ shape = {}",M_.shape)
    return 1.0 + jnp.sum(M*M_,axis=1)
