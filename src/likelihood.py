import jax.numpy as jnp
import jax

def truncate_and_sum(A, v):
    # A has shape (P, K)
    # v is a vector of length P with indices indicating where to truncate
    # jax.debug.print("A={}",A)
    _, K = A.shape
    
    # Create a mask where for each row i, only columns up to v[i] are included
    mask = jnp.arange(K) < v[:, None]  # Broadcasting over rows to create the mask
    # jax.debug.print("v={}",v)
    # Apply the mask to A
    masked_A = A * mask

    # Sum the masked rows
    row_sums = jnp.sum(masked_A, axis=1)

    return row_sums

@jax.jit
def compute_ell1(obs_mean, obs_var, legQuadwts):
    obs_mean_link = jnp.exp(obs_mean + 0.5*obs_var)
    return jnp.einsum('ijk,jk->ij',obs_mean_link,legQuadwts)
