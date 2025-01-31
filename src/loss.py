import jax.numpy as jnp
import jax
from src.kernel import precompute_Kzz
from src.observations import compute_obs_means,compute_obs_vars
from src.likelihood import truncate_and_sum, compute_ell1
from src.kldivergence import vmap_kl_latents
from src.cholesky import build_covs_from_cholvecs
from functools import partial
import pprint

@partial(jax.vmap, in_axes = (None,1),out_axes = 1)
def compute_mean_factor(Kzz_cho, vMean):
    return jax.scipy.linalg.cho_solve((Kzz_cho,True), vMean)

def construct_C_pytree(C, dummy_dict):
    C_tree = jax.tree.map(lambda key: C[key][None,:], dummy_dict)
    return C_tree

def construct_d_pytree(d, dummy_dict):
    d_tree = jax.tree.map(lambda key: d[key], dummy_dict)
    return d_tree

def ensure_2d(arr):
    return jnp.expand_dims(arr, axis=-1) if arr.ndim == 1 else arr

def compute_elbo(params, spike_times, quad, trunc_indices, dummy_dict):
    
    Kzz = precompute_Kzz(params['ind_points_locs'],params['kernel_params'])
    Kzz_cho = jax.scipy.linalg.cholesky(Kzz, lower=True)
    mean_factor = compute_mean_factor(Kzz_cho,params['vMean'])
    vCov = build_covs_from_cholvecs(params['vChol'])

    Cmat = params['C']
    dvec = params['d']
    dvec = dvec[:,:,None]

    C = construct_C_pytree(Cmat, dummy_dict)
    d = construct_d_pytree(params['d'], dummy_dict)
    obs_mean_func = lambda x, y, z : compute_obs_means(params['ind_points_locs'], params['kernel_params'], mean_factor, x, y, z)  
    obs_means_sp = jax.tree.map(obs_mean_func,spike_times,C,d)
    obs_means_sp = jax.tree.map(ensure_2d, obs_means_sp)
    ell2 = jax.tree.map(lambda x, y: truncate_and_sum(x, y), obs_means_sp, trunc_indices)
    ell2 = jnp.stack(jax.tree.leaves(ell2),axis=0)
    
    obs_mean_quad = compute_obs_means(params['ind_points_locs'], params['kernel_params'], mean_factor, quad['points'], Cmat, dvec)
    obs_var_quad =  compute_obs_vars(params['ind_points_locs'], params['kernel_params'], Kzz, Kzz_cho, quad['points'], vCov, Cmat)   
    ell1 = compute_ell1(obs_mean_quad, obs_var_quad, quad['weights'])
    loglik = jnp.sum(-ell1.reshape(-1)) + jnp.sum(ell2)
    kld = vmap_kl_latents(Kzz, Kzz_cho, params['vMean'], vCov)
    
    return -1*(loglik - jnp.sum(kld))

def compute_elbo_fixedZ(params, spike_times, quad, ind_points_locs, trunc_indices, dummy_dict):
    
    Kzz = precompute_Kzz(ind_points_locs,params['kernel_params'])
    Kzz_cho = jax.scipy.linalg.cholesky(Kzz, lower=True)
    mean_factor = compute_mean_factor(Kzz_cho,params['vMean'])
    vCov = build_covs_from_cholvecs(params['vChol'])

    Cmat = params['C']
    dvec = params['d']
    dvec = dvec[:,:,None]

    C = construct_C_pytree(Cmat, dummy_dict)
    d = construct_d_pytree(params['d'], dummy_dict)
    obs_mean_func = lambda x, y, z : compute_obs_means(ind_points_locs, params['kernel_params'], mean_factor, x, y, z)  
    obs_means_sp = jax.tree.map(obs_mean_func,spike_times,C,d)
    obs_means_sp = jax.tree.map(ensure_2d, obs_means_sp)

    ell2 = jax.tree.map(lambda x, y: truncate_and_sum(x, y), obs_means_sp, trunc_indices)
    ell2 = jnp.stack(jax.tree.leaves(ell2),axis=0)
    
    obs_mean_quad = compute_obs_means(ind_points_locs, params['kernel_params'], mean_factor, quad['points'], Cmat, dvec)
    obs_var_quad =  compute_obs_vars(ind_points_locs, params['kernel_params'], Kzz, Kzz_cho, quad['points'], vCov, Cmat)   
    ell1 = compute_ell1(obs_mean_quad, obs_var_quad, quad['weights'])
    kld = vmap_kl_latents(Kzz, Kzz_cho, params['vMean'], vCov)
    log_lik = jnp.sum(-ell1.reshape(-1)) + jnp.sum(ell2)
    return -1*(log_lik - jnp.sum(kld))

def compute_elbo_twoarea(params, spike_times, quad, trunc_indices, dummy_dict):
    
    Kzz = precompute_Kzz(params['ind_points_locs'],params['kernel_params'])
    Kzz_cho = jax.scipy.linalg.cholesky(Kzz, lower = True)
    mean_factor = compute_mean_factor(Kzz_cho,params['vMean'])
    vCov = build_covs_from_cholvecs(params['vChol'])
    C1 = params['C1']
    C2 = params['C2']
    Cmat = jax.scipy.linalg.block_diag(C1,C2)
    C = construct_C_pytree(Cmat, dummy_dict)
    d = construct_d_pytree(params['d'], dummy_dict)

    obs_mean_func = lambda x, y, z : compute_obs_means(params['ind_points_locs'], params['kernel_params'], mean_factor, x, y, z)  
    obs_means_sp = jax.tree.map(obs_mean_func,spike_times,C,d)
    ell2 = jax.tree.map(lambda x, y: truncate_and_sum(x, y), obs_means_sp, trunc_indices)
    ell2 = jnp.stack(jax.tree.leaves(ell2),axis=0)
    dvec = params['d'][:,:,None]
    obs_mean_quad = compute_obs_means(params['ind_points_locs'], params['kernel_params'], mean_factor, quad['points'], Cmat, dvec)
    obs_var_quad =  compute_obs_vars(params['ind_points_locs'], params['kernel_params'], Kzz, Kzz_cho, quad['points'], vCov, Cmat)   
    
    ell1 = compute_ell1(obs_mean_quad, obs_var_quad, quad['weights'])
    kld = vmap_kl_latents(Kzz, Kzz_cho, params['vMean'], vCov)

    return -1*(jnp.sum(-ell1.reshape(-1)) + jnp.sum(ell2) - jnp.sum(kld))

def compute_elbo_twoarea_fixedZ(params, spike_times, quad, ind_points_locs, trunc_indices, dummy_dict):
    
    Kzz = precompute_Kzz(ind_points_locs,params['kernel_params'])
    Kzz_cho = jax.scipy.linalg.cholesky(Kzz, lower = True)
    mean_factor = compute_mean_factor(Kzz_cho,params['vMean'])
    vCov = build_covs_from_cholvecs(params['vChol'])
    C1 = params['C1']
    C2 = params['C2']
    Cmat = jax.scipy.linalg.block_diag(C1,C2)
    C = construct_C_pytree(Cmat, dummy_dict)
    d = construct_d_pytree(params['d'], dummy_dict)

    obs_mean_func = lambda x, y, z : compute_obs_means(ind_points_locs, params['kernel_params'], mean_factor, x, y, z)  
    obs_means_sp = jax.tree.map(obs_mean_func,spike_times,C,d)
    ell2 = jax.tree.map(lambda x, y: truncate_and_sum(x, y), obs_means_sp, trunc_indices)
    ell2 = jnp.stack(jax.tree.leaves(ell2),axis=0)
    dvec = params['d'][:,:,None]
    obs_mean_quad = compute_obs_means(ind_points_locs, params['kernel_params'], mean_factor, quad['points'], Cmat, dvec)
    obs_var_quad =  compute_obs_vars(ind_points_locs, params['kernel_params'], Kzz, Kzz_cho, quad['points'], vCov, Cmat)   
    
    ell1 = compute_ell1(obs_mean_quad, obs_var_quad, quad['weights'])
    kld = vmap_kl_latents(Kzz, Kzz_cho, params['vMean'], vCov)

    return -1*(jnp.sum(-ell1.reshape(-1)) + jnp.sum(ell2) - jnp.sum(kld))

def objective_fn(spike_times, quad, trunc_indices,dummy_dict):
    return jax.jit(lambda x: compute_elbo(x,spike_times,quad,trunc_indices,dummy_dict))

def objective_fn_fixedZ(spike_times, quad, ind_points_locs, trunc_indices,dummy_dict):
    return jax.jit(lambda x: compute_elbo_fixedZ(x,spike_times,quad,ind_points_locs,trunc_indices,dummy_dict))
    # return lambda x: compute_elbo_fixedZ(x,spike_times,quad,ind_points_locs,trunc_indices,dummy_dict)  

def objective_fn_twoarea(spike_times, quad, trunc_indices, dummy_dict):
    return jax.jit(lambda x: compute_elbo_twoarea(x,spike_times,quad,trunc_indices, dummy_dict))

def objective_fn_twoarea_fixedZ(spike_times, quad, ind_points_locs, trunc_indices, dummy_dict):
    return jax.jit(lambda x: compute_elbo_twoarea_fixedZ(x,spike_times,quad, ind_points_locs, trunc_indices, dummy_dict))


