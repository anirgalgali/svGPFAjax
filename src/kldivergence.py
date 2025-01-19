import jax.numpy as jnp
from jax import vmap
import jax


def compute_kl_divergence(Kzz, Kzz_cho, vMean, vCov):

    d = Kzz_cho.shape[-1]
    ess = vCov + jnp.outer(vMean, vMean)
    _, Kzz_logdet = jnp.linalg.slogdet(Kzz)
    _, vCov_logdet = jnp.linalg.slogdet(vCov)
    solve_term = jax.scipy.linalg.cho_solve((Kzz_cho,True),ess)
    trace_term = jnp.trace(solve_term)
    kl = 0.5*(trace_term + Kzz_logdet - vCov_logdet - d)
    return kl

vmap_kl_trials = vmap(compute_kl_divergence,in_axes=(None,None,0,0))
vmap_kl_latents = vmap(vmap_kl_trials, in_axes=(0,0,0,0))