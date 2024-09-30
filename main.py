import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
import jax.numpy as jnp
import numpy as np
import jax
from loss import objective_fn,objective_fn_twoarea,objective_fn_fixedZ,objective_fn_twoarea_fixedZ
from cholesky import vmapped_lower_triangular
from quadrature import getLegQuadPointsAndWeights
import jaxopt
import time
import matplotlib.pyplot as plt
import math
import pickle
import subprocess as sp
import random
import json
import sys
import argparse
import pprint

def get_gpu_memory():
    print(sp.check_output("nvidia-smi").decode('ascii'))

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
    key, subkey1, subkey2, subkey3= jax.random.split(key, 4)
    C1 = jax.random.normal(key = subkey1, shape = (n_neurons[0], n_latents[0]))*0.1
    C2 = jax.random.normal(key = subkey2, shape = (n_neurons[1], n_latents[1]))*0.1
    d =  jax.random.normal(key = subkey3, shape = (sum(n_neurons), 1))*0.1
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
    ind_points_locs = jnp.linspace(0, T, num = n_ind_points)
    ind_points_locs = jnp.repeat(ind_points_locs[jnp.newaxis,:], n_latents, axis=0)
    return ind_points_locs

def initialize_kernel_params(n_latents,sigma=1):
    return sigma * jnp.ones((n_latents,))

def initialize_variational_params(key, n_ind_points, n_latents, n_trials):
    key,subkey = jax.random.split(key)
    vMean = jax.random.normal(key = subkey, shape = (n_latents,n_trials,n_ind_points,))
    vCov = 1e-2*jnp.eye(n_ind_points)
    vCov = vCov[jnp.newaxis,jnp.newaxis,...]
    vCov = jnp.repeat(vCov,repeats=n_latents,axis=0)
    vCov = jnp.repeat(vCov,repeats=n_trials,axis=1)
    vChol = jax.scipy.linalg.cholesky(vCov,lower=True)
    vChol = vmapped_lower_triangular(vChol)
    return vMean, vChol, key

def subset_trials_ids_data(selected_trials_ids, trials_ids, spikes_times,
                           trials_start_times, trials_end_times):
    indices = np.nonzero(np.in1d(trials_ids, selected_trials_ids))[0]
    spikes_times_subset = [spikes_times[i] for i in indices]
    trials_start_times_subset = [trials_start_times[i] for i in indices]
    trials_end_times_subset = [trials_end_times[i] for i in indices]

    return spikes_times_subset, trials_start_times_subset, trials_end_times_subset

def subset_clusters_ids_data(selected_clusters_ids, clusters_ids,
                             spikes_times, unit_probes_index):
    indices = np.nonzero(np.in1d(clusters_ids, selected_clusters_ids))[0]
    n_trials = len(spikes_times)
    spikes_times_subset = [[spikes_times[r][i] for i in indices]
                           for r in range(n_trials)]
    unit_probes_subset = [unit_probes_index[i] for i in indices]

    return spikes_times_subset, unit_probes_subset


def process_data_for_fit(data_filename, data_filter_pars):
    
    with open(data_filename, "rb") as f:
        load_res = pickle.load(f)
    
    spikes_times = load_res["spikes_times"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]
    unit_probes_index = load_res["unit_probes_index"]

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials_ids = np.arange(n_trials)
    neurons_indices = np.arange(n_neurons).tolist()


    if "clusters_ids_filename" in data_filter_pars.keys():
        selected_clusters_ids = np.genfromtxt(data_filter_pars["clusters_ids_filename"],dtype=np.uint64)
        clusters_ids = np.arange(n_neurons)
        spikes_times, unit_probes_index = subset_clusters_ids_data(
            selected_clusters_ids=selected_clusters_ids,
            clusters_ids=clusters_ids,
            spikes_times=spikes_times, unit_probes_index=unit_probes_index)

    if "trials_ids_filename" in data_filter_pars.keys():
        selected_trials_ids = np.genfromtxt(data_filter_pars["trials_ids_filename"] , dtype=np.uint64)
        spikes_times, trials_start_times, trials_end_times = \
            subset_trials_ids_data(
                selected_trials_ids=selected_trials_ids,
                trials_ids=trials_ids,
                spikes_times=spikes_times,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)

    
    if data_filter_pars["data_fraction_fit"] < 1.0:
        selected_inds = np.random.choice(len(spikes_times),size = int(np.round(data_filter_pars["data_fraction_fit"]*len(spikes_times))),replace = False)
        selected_inds = list(selected_inds)
        print(selected_inds)
        spikes_times = [spikes_times[i] for i in selected_inds]
        trials_start_times = [trials_start_times[i] for i in selected_inds]
        trials_end_times = [trials_end_times[i] for i in selected_inds]
        trials_ids = [trials_ids[i] for i in selected_inds]

    elif data_filter_pars["data_fraction_fit"] > 1.0:
        raise ValueError('invalid value')
    
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    if "clusters_ids_filename" in data_filter_pars.keys():
        clusters_ids = selected_clusters_ids
    else:
        clusters_ids = neurons_indices

    spikes_times_array, trunc_indices = preprocess_data(spikes_times)
    return spikes_times_array, trials_start_times, trials_end_times, trunc_indices, unit_probes_index, trials_ids, clusters_ids

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_fit_ind_points_locs", help ="use this option if you want optimize the inducing point locations",
                        action="store_true")
    
    parser.add_argument("--do_fit_blocked", help ="use this option if you want fit a structured (block-diagonal) observation matrix",
                        action="store_true")
    
    parser.add_argument("--data_fraction_fit", help ="fraction of data to fit",
                    type=float, default = 1.0)
    
    parser.add_argument("--n_latents_1", help="number of latent processes in area 1",
                        type=int, default=5)
    
    parser.add_argument("--n_latents_2", help="number of latent processes in area 2",
                        type=int, default=5)
    
    parser.add_argument("--n_quad", help="number of points for quadrature",
                        type=int, default=100)
    
    parser.add_argument("--n_ind_points",
                        help="common number of inducing points",
                        type=int, default=10)
    
    parser.add_argument("--data_filename_pattern",
                        help="data filename pattern",
                        type=str,
                        default="physical_switches_static_triplets")
    
    parser.add_argument("--chunk_size",help = "indicates which datafile with chunk size property to analyze",
                        type = int, default = 1)
    
    parser.add_argument("--clusters_ids_filename", help="clusters ids filename",
                        type=str, default="../svGPFA_rivarly/init/neuronIDs_thresholded_common_union.csv")
    
    parser.add_argument("--max_em_iter", help = "maximum number of EM iterations",type=int,default=100000)

    parser.add_argument("--scale_fac", help = "scaling factor for kernel length scale (usually between 0.80 and 1.60)",type=float,default=1.0)

    parser.add_argument("--seed", help = "random seed for fit", type = int, default = 0)

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    if args.seed == 0:
        seed = np.random.randint(low = 0, high = 1000000)
    else:
        seed = args.seed

    key = jax.random.PRNGKey(seed)
    save_file_id = random.randint(0, 10**8)

    buffer_size = 10
    tolerance = 1e-1
    
    threshold_count_iter = 5
    
    n_latents = [args.n_latents_1, args.n_latents_2]
    n_total_latents = sum(n_latents)

    optim_params = dict(tol = 1e-6, 
                    max_stepsize=1.0, 
                    jit = True
                    )

    unit_selection_strategy = os.path.splitext(os.path.basename(args.clusters_ids_filename))[0]
    unit_selection_strategy = unit_selection_strategy.split('_')[-1]

    if args.chunk_size != 1:
        data_filename = f'../svGPFA_rivarly/data/epochedSpikes_Alfie_231024_200742_{args.data_filename_pattern:s}_chunksize={str(args.chunk_size)}.pickle'
        model_save_filename = f'./results/{save_file_id}_seed={seed}_{unit_selection_strategy}_{args.data_filename_pattern:s}_chunksize={str(args.chunk_size)}.pickle'
        model_metadata_filename = f'./results/{save_file_id}_seed={seed}_{unit_selection_strategy}_{args.data_filename_pattern:s}_chunksize={str(args.chunk_size)}.json'
    else:
        data_filename = f'../svGPFA_rivarly/data/epochedSpikes_Alfie_231024_200742_{args.data_filename_pattern:s}.pickle'
        model_save_filename = f'./results/{save_file_id}_seed={seed}_{unit_selection_strategy}_{args.data_filename_pattern:s}.pickle'
        model_metadata_filename = f'./results/{save_file_id}_seed={seed}_{unit_selection_strategy}_{args.data_filename_pattern:s}.json'
    
    data_filter_params = dict()
    data_filter_params["data_fraction_fit"] = args.data_fraction_fit
    data_filter_params["clusters_ids_filename"] = args.clusters_ids_filename

    # get spike_times
    spikes_times_array, trials_start_times, trials_end_times, trunc_indices, unit_probes_index, trials_ids, clusters_ids = process_data_for_fit(data_filename, data_filter_params)

    if jax.default_backend() == "gpu":
        get_gpu_memory()

    n_neurons = []
    _, n_neurons = np.unique(unit_probes_index,return_counts = True)
    max_trial_length =  np.amax(trials_end_times)
    n_trials = len(trials_end_times)

    quad = dict()
    quad['points'], quad['weights'] = getLegQuadPointsAndWeights(args.n_quad, np.zeros((n_trials ,)),\
                                                    max_trial_length*np.ones((n_trials ,)))
    
    params0 = dict()

    params0['kernel_params'] = initialize_kernel_params(n_total_latents,sigma = args. scale_fac * (max_trial_length/args.n_ind_points))
    params0['vMean'], params0['vChol'], key  = initialize_variational_params(key,args.n_ind_points, n_total_latents, n_trials)

    if args.do_fit_ind_points_locs:
        params0['ind_points_locs'] = initialize_inducing_points(max_trial_length, args.n_ind_points, n_total_latents)
    else:
        fixed_ind_points_locs = initialize_inducing_points(max_trial_length, args.n_ind_points, n_total_latents)

    if args.do_fit_blocked:
        params0['C1'],params0['C2'], params0['d'], dummy_dict, key = initialize_observationmodel_params_blocked(key,n_neurons,n_latents)

    else:
        params0['C'], params0['d'], dummy_dict, key = initialize_observationmodel_params(key,sum(n_neurons),n_total_latents)

    if args.do_fit_blocked:
        if args.do_fit_ind_points_locs:
            free_energy = objective_fn_twoarea(spikes_times_array, quad, trunc_indices, dummy_dict)
        else:
            free_energy = objective_fn_twoarea_fixedZ(spikes_times_array, quad, fixed_ind_points_locs,trunc_indices, dummy_dict)
    else:
        if args.do_fit_ind_points_locs:
            free_energy = objective_fn(spikes_times_array, quad, trunc_indices, dummy_dict)
        else:
            free_energy = objective_fn_fixedZ(spikes_times_array, quad, fixed_ind_points_locs,trunc_indices, dummy_dict)

    solver = jaxopt.LBFGS(fun= free_energy, **optim_params)
    print(f'Initializing LBFGS solver ....')

    optim_state = solver.init_state(params0)
    print(f'Completed Initializing LBFGS solver')

    if jax.default_backend() == "gpu":
        get_gpu_memory()

    params = params0
    lb = -free_energy(params0)
    print(f'ElBO at initialization ={lb}')
    lower_bound_hist = [lb]
    elapsed_time_hist = []
    start_time = time.time()

    stop_counter = 0
    running_mean = 0.0
    for step in range(args.max_em_iter):
        params, optim_state = solver.update(params= params, state=optim_state)
        lb = -optim_state.value.item()
        lower_bound_hist.append(lb)
        elapsed_time_hist.append(time.time() - start_time)

        if step < buffer_size:
            print(f"Iteration {step} - Lower Bound: {lb}")

        if step >= buffer_size:
            
            running_mean_new = sum(lower_bound_hist[-buffer_size:])/buffer_size
            delta = np.abs(running_mean_new - running_mean)
            print(f"Iteration {step} - Lower Bound: {lb}, Mov. Avg : {running_mean_new}, delta: {delta}")  

            if delta <= tolerance:
                stop_counter += 1
               
            else:
                stop_counter = 0
            
            running_mean = running_mean_new  # Reset the counter if the value changes

            if stop_counter >= threshold_count_iter:
                print(f"Terminating after {step} iterations due to no change in ELBO.")
                break

        if step % 500 == 0:
            if jax.default_backend() == "gpu":
                get_gpu_memory()

    elapsed_time = time.time() - start_time
    print(f"elapsed time={elapsed_time}")

    result = {"params":params, "state":optim_state, "elapsed_time_hist": elapsed_time_hist, "lower_bound_hist": lower_bound_hist}
    with open( model_save_filename, "wb") as f:
        pickle.dump(result, f)
        print("Saved results to {:s}".format(model_save_filename))

    # save estimated config
    estim_res_config = {
        "trials_ids": trials_ids.tolist(),
        "neurons_indices": clusters_ids.tolist(),
        "do_fit_ind_points_locs": args.do_fit_ind_points_locs,
        "do_fit_blocked": args.do_fit_blocked,
        "n_latents_1": args.n_latents_1,
        "n_latents_2": args.n_latents_2,
        "n_quad": args.n_quad,
        "common_n_ind_points": args.n_ind_points,
        "neuron_count_per_area":n_neurons.tolist(),
        "max_trial_duration": max_trial_length,
        "epoched_spikes_times_filename": data_filename,
        "data_fraction_fit": data_filter_params["data_fraction_fit"],
        "max_em_iter": args.max_em_iter,
        "seed": seed,
        "fixed_ind_points_locs": None if args.do_fit_ind_points_locs else fixed_ind_points_locs.tolist(),
        "kernel_scale_fac": args.scale_fac
    }

    with open(model_metadata_filename, 'w') as f:
        json.dump(estim_res_config, f)



if __name__ == '__main__':
    
    main(sys.argv)