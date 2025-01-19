import math
import jax.numpy as jnp
import jax
from jax import vmap
import jax.lax as lax

def set_lower_triangular(v):
    size = v.size
    n = (math.sqrt(8*size + 1) - 1)//2
    n = int(n)
    tril_indices = jnp.tril_indices(n)
    chol_matrix = jnp.zeros((n, n))
    chol_matrix = chol_matrix.at[tril_indices].set(v)
    return chol_matrix

def build_covs_from_cholvecs(chol_vecs):
    # """Build covariances from vector respresntations of their Cholesky
    # descompositions.

    # :param chol_vecs: vector respresentations of the lower-triangular Cholesky factors
    # :type  jax.Array \in (n_latents, n_trials, Pk, 1) where Pk=(n_ind_points[k] * (n_ind_points[k] + 1)) / 2
    # """
    # Pk = (n_ind_points * (n_ind_points + 1)) / 2 then
    # n_ind_points = (math.sqrt(1 + 8 * M) - 1) / 2
    # n_latents = chol_vecs.shape[0]
    # n_trials = chol_vecs.shape[1]

    get_lower_cholesky = jax.vmap(jax.vmap(set_lower_triangular))
    chols = get_lower_cholesky(chol_vecs)
    covs = jnp.matmul(chols, jnp.transpose(chols, (0, 1, 3, 2)))
    return covs

def get_lower_triangular(X):
    dim = X.shape[-1]
    X_tril = jnp.tril(X)
    mask = jnp.tri(dim,dtype= jnp.bool)
    return jnp.ravel(X_tril[mask])

vmapped_lower_triangular = vmap(vmap(get_lower_triangular))