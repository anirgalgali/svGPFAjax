import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
import jax.numpy as jnp
import numpy as np
import jax
from loss import objective_fn, objective_fn_twoarea
from cholesky import vmapped_lower_triangular
from quadrature import getLegQuadPointsAndWeights
import jaxopt
import time
import matplotlib.pyplot as plt
import math
import pickle
import subprocess as sp
from collections import OrderedDict
import scipy

seed = np.random.randint(low = 42, high = 450000)
key = jax.random.PRNGKey(seed)

rng = np.random.default_rng(42)

def get_gpu_memory():
    print(sp.check_output("nvidia-smi").decode('ascii'))

def simulate_spikes_from_rates(rates, T, n_trials, ngrid):

    def sample_inhomogenous(rate, T, ngrid):
        x = np.linspace(0,T,num=ngrid)
        lam0 = np.amax(rate(x))
        u = rng.uniform(size=(np.ceil(1.5*T*lam0).astype(np.int64),))
        x = np.cumsum(-(1/lam0)* np.log(u))
        x = x[x<T]
        n=sum(x<T)
        l = rate(x)
        x = x[rng.uniform(size=(n,)) < l/lam0]
        return x

    n_neurons = len(rates)
    sps = []
    for m in range(n_trials):
        sps_m = []
        for i in range(n_neurons):
            sps_m.append(sample_inhomogenous(rates[i], T, ngrid))
        
        sps.append(sps_m)

    return sps

def preprocess_data(spikes):

    spike_times = dict()
    trunc_indices = dict()
    max_spikes = []
    n_spikes = []
    n_trials = len(spikes)
    n_neurons = len(spikes[0])
    for i in range(n_neurons):
        n_spikes_i = []
        for j in range(n_trials):
            n_spikes_i.append(len(spikes[j][i]))
        max_spikes.append(np.amax(np.asarray(n_spikes_i)))
        n_spikes.append(n_spikes_i)

    for i in range(n_neurons):

        spike_time_matrix = np.empty((n_trials,max_spikes[i]))
        spike_time_matrix[:] = np.nan
        
        for j in range(n_trials):
            spike_time_matrix[j,:n_spikes[i][j]] = spikes[j][i]

        spike_times[f'Unit-{i+1}'] = jnp.asarray(spike_time_matrix)
        trunc_indices[f'Unit-{i+1}'] = jnp.asarray(n_spikes[i])

    pad_fn = lambda x: jnp.nan_to_num(x,nan=1e9)
    spike_times = {k:pad_fn(v) for k, v in spike_times.items()}
    return spike_times, trunc_indices

def initialize_observationmodel_params_blocked(key, n_neurons, n_latents):
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    C1 = jax.random.normal(key = subkey1, shape = (n_neurons[0], n_latents[0]))*0.1
    C2 = jax.random.normal(key = subkey2, shape = (n_neurons[1], n_latents[1]))*0.1
    d = jax.random.normal(key = subkey3, shape = (n_neurons[0] + n_neurons[1], 1))*0.1
    dummy_dict = dict()
    for n in range(sum(n_neurons)):
        dummy_dict[f'Unit-{n+1}'] = jnp.asarray(n)
    return C1, C2, d, dummy_dict, key

def initialize_observationmodel_params(key, n_neurons, n_latents):
    key, subkey1, subkey2 = jax.random.split(key, 3)
    C = jax.random.normal(key = subkey1, shape = (n_neurons, n_latents))*0.1
    d = jax.random.normal(key = subkey2, shape = (n_neurons, 1))*0.1
    dummy_dict = dict()
    for n in range(n_neurons):
        dummy_dict[f'Unit-{n+1}'] = jnp.asarray(n)
    return C, d, dummy_dict, key       

def initialize_inducing_points(T, n_ind_points ,n_latents):
    ind_points_locs = jnp.linspace(0, T, n_ind_points)
    ind_points_locs = jnp.repeat(ind_points_locs[jnp.newaxis,:], n_latents, axis=0)
    return ind_points_locs

def initialize_kernel_params(n_latents,sigma=1):
    return sigma * jnp.ones((n_latents,))

def initialize_variational_params(key, n_ind_points, n_latents, n_trials):
    key, subkey = jax.random.split(key)
    vMean = jax.random.normal(key = subkey, shape = (n_latents,n_trials,n_ind_points,))
    vCov = 1e-2*jnp.eye(n_ind_points)
    vCov = vCov[jnp.newaxis,jnp.newaxis,...]
    vCov = jnp.repeat(vCov,repeats=n_latents,axis=0)
    vCov = jnp.repeat(vCov,repeats=n_trials,axis=1)
    vChol = jax.scipy.linalg.cholesky(vCov,lower=True)
    vChol = vmapped_lower_triangular(vChol)
    return vMean, vChol, key

if __name__ == '__main__':

    MAX_EM_ITER = 5000
    do_fit_block = True
    runid = 11
    if do_fit_block:
        modelSaveFilename = f'./results/demo2Dlatents_twoareasimfit_run00{runid:d}_estimatedModel.pickle'
    else:
        modelSaveFilename = f'./results/demo2Dlatents_twoareasim_oneareafit_run00{runid:d}_estimatedModel.pickle'


    fit_params = dict(n_latents_1 = 2,
                      n_latents_2 = 2,
                      n_ind_points = 10,
                      n_quad = 100)

    data_params = dict(n_neurons_1 = 35,
                       n_neurons_2 = 45,
                       n_trials = 50,
                       max_trial_length = 20,
                       n_grid = 2000)
    
    optim_params = dict(tol = 1e-6,
                        maxiter = MAX_EM_ITER, 
                        max_stepsize=1.0, 
                        jit = True
                        )

    n_total_neurons = data_params['n_neurons_1'] + data_params['n_neurons_2']
    n_total_latents = fit_params['n_latents_1'] + fit_params['n_latents_2']
    n_neurons = [data_params['n_neurons_1'], data_params['n_neurons_2']]
    n_latents = [fit_params['n_latents_1'], fit_params['n_latents_2']]

    latent_func_1 = lambda t: 1.5 * np.exp(-0.5 *((t-10) ** 2 / 12)) * np.sin(2 * math.pi * (1/12) * t)
    latent_func_2 = lambda t: 0.8 * np.exp(-0.5*((t-5) ** 2/8)) * np.cos(2 * math.pi * .12 * t )  + 0.9 * np.exp(-0.5 * ((t-10)**2/12)) * np.sin(2 * math.pi * t * 0.1 + 1.5)

    latent_func_3 = lambda t: 0.9 * np.exp(-0.5*((t-12) ** 1/6)) * np.sin(2 * math.pi * (1/12) * t + 1.0)
    latent_func_4 = lambda t: 0.8 * np.exp(-0.5*((t-2) ** 2/4)) * np.cos(2 * math.pi * .12 * t + 0.3 )  + 0.2 * np.exp(-0.5 * ((t-8)**2/12)) * np.sin(2 * math.pi * t * 0.1 - 0.5)


    trueC1 = rng.normal(size= (data_params['n_neurons_1'],fit_params['n_latents_1']))
    trueC1 = trueC1 * np.asarray([[1.2,1.3]])
    trueC1 = 0.4*trueC1

    trueC2 = rng.normal(size = (data_params['n_neurons_2'],fit_params['n_latents_2']))
    trueC2 = trueC2 * np.asarray([[1.2,1.3]])
    trueC2 = 0.4*trueC2

    trueC = scipy.linalg.block_diag(trueC1,trueC2)

    trued = -0.1*np.ones((n_total_neurons,1))
    
    # get_gpu_memory()

    rate_functions = []
    for n in range(n_total_neurons):
        rate_functions.append(lambda t, n=n: np.exp(np.matmul(trueC[n,:], np.stack([latent_func_1(t), latent_func_2(t),latent_func_3(t),latent_func_4(t)])) + trued[n]))

    spikes = simulate_spikes_from_rates(rate_functions, data_params['max_trial_length'], data_params['n_trials'], data_params['n_grid'])    


    # get_gpu_memory()

    spike_times, trunc_indices = preprocess_data(spikes)

    # get_gpu_memory()


    params0 = dict()
    params0['kernel_params'] = initialize_kernel_params(n_total_latents)

    quad = dict()
    quad['points'], quad['weights'] = getLegQuadPointsAndWeights(fit_params['n_quad'], \
                                                    np.zeros((data_params['n_trials'],)),\
                                                    data_params['max_trial_length']*np.ones((data_params['n_trials'],)))
  
    params0['vMean'], params0['vChol'], key  = initialize_variational_params(key,fit_params['n_ind_points'],n_total_latents,data_params['n_trials'])

    if do_fit_block:
        params0['C1'],params0['C2'],params0['d'],dummy_dict,key = initialize_observationmodel_params_blocked(key,n_neurons,n_latents)
        free_energy = objective_fn_twoarea(spike_times, quad, trunc_indices, dummy_dict)
    else:
        params0['C'], params0['d'],dummy_dict, key = initialize_observationmodel_params(key,n_total_neurons,n_total_latents)
        free_energy = objective_fn(spike_times, quad, trunc_indices, dummy_dict)

    params0['ind_points_locs'] = initialize_inducing_points(data_params['max_trial_length'],fit_params['n_ind_points'],n_total_latents)
    
    start_time = time.time()

    solver = jaxopt.LBFGS(fun= free_energy, **optim_params)
    print(f'Initializing LBFGS solver ....')
    # get_gpu_memory()
    optim_state = solver.init_state(params0)
    # get_gpu_memory()
    print(f'Completed Initializing LBFGS solver')
    params = params0
    lb = -free_energy(params0)
    print(f'ElBO at initialization ={lb}')
    lower_bound_hist = [lb]
    elapsed_time_hist = []
    start_time_iter = time.time()
    # running_mean = 0.0

    for step in range(MAX_EM_ITER):
        params, optim_state = solver.update(params= params, state=optim_state)
        lb = -optim_state.value.item()
        lower_bound_hist.append(lb)
        elapsed_time_hist.append(time.time() - start_time_iter)
        print(f"Iteration {step}: {lb}")

    elapsed_time = time.time() - start_time
    print(f"elapsed time={elapsed_time}")
    result = {"params":params, "state":optim_state, "elapsed_time_hist": elapsed_time_hist, "lower_bound_hist": lower_bound_hist, "seed":seed, "fit_type": do_fit_block}
    with open(modelSaveFilename, "wb") as f:
        pickle.dump(result, f)
        print("Saved results to {:s}".format(modelSaveFilename))