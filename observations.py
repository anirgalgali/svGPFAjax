import jax.numpy as jnp
import jax
from latents import compute_latent_means, compute_latent_variances

@jax.jit
def compute_obs_means(ind_points_locs, kernel_param, meanFactor, times, loadings, offset):
    # jax.debug.print("loadings shape {}", loadings.shape)
    # jax.debug.print("offset shape {}", offset.shape)
    return (jnp.einsum('ij,jkl->ikl',loadings,compute_latent_means(kernel_param, ind_points_locs, meanFactor, times)) + offset).squeeze()

@jax.jit
def compute_obs_vars(ind_points_locs, kernel_param, Kzz, Kzz_cho, times, vCov, loadings):
    return jnp.einsum('ij,jkl->ikl',loadings**2,compute_latent_variances(kernel_param, ind_points_locs, Kzz, Kzz_cho, vCov,times))
