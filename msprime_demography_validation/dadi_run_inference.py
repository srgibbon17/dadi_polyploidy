"""
Define a set of demographic models to exactly mirror the 
msprime models defined in generate_msprime_samples.py.

Command line usage:
python3 path_to_script/dadi_run_inference.py \
    --config-file path_to_config_file/inference_configs_ex.yaml \
    --input-file path_to_tree_sequence_file/input.trees

See the inference_configs_ex.yaml file for an example of the config file.
"""

# note: all of the allotetraploid models considered here include a divergence period 
# between the two diploid populations for T=1 diffusion units. 

import dadi
import dadi.Polyploidy.Integration as PolyInt
import numpy as np
import nlopt
import os
import tskit
import yaml
import argparse
import time

# utility functions for calculating spectra from tree sequences
def get_sfs_from_ts(ts):
    """
    Function to compute the SFS from an arbitrary tree sequence.

    ts: a tskit tree sequence object 

    Returns:
    sfs: a dadi Spectrum object from the tskit tree sequence
    """
    pop_ids = ts.individuals_population
    node_ids = ts.individuals_nodes

    unique_pops = np.unique(pop_ids)

    sample_sets = []
    
    for pop_id in unique_pops:
        # get the indices of individuals in this population
        inds_in_pop = np.where(pop_ids == pop_id)[0]
        # get the nodes of these individuals
        nodes_in_pop = node_ids[inds_in_pop]
        # then, turn the nodes into a sample set
        sample_set = np.unique(nodes_in_pop)
        sample_sets.append(sample_set)

    # then, use those sample sets to compute the SFS
    sfs = ts.allele_frequency_spectrum(sample_sets = sample_sets, polarised=True, span_normalise=False)
    sfs = dadi.Spectrum(sfs)
    return sfs 

def get_bootstrap_spectra(ts, num_windows, num_bootstraps, seed):
    """
    Calculate a single boostrap SFS from a tree sequence.

    ts: a tskit tree sequence object
    num_windows: number of windows to split the genome into
    num_bootstraps: number of bootstrap spectra to generate
    seed: random seed for reproducibility (for sampling from the genome chunks)

    Returns:
    bootstrap_spectra: a list of bootstrap spectra (dadi.Spectrum objects)
    """
    pop_ids = ts.individuals_population
    node_ids = ts.individuals_nodes

    unique_pops = np.unique(pop_ids)

    sample_sets = []
    
    for pop_id in unique_pops:
        # get the indices of individuals in this population
        inds_in_pop = np.where(pop_ids == pop_id)[0]
        # get the nodes of these individuals
        nodes_in_pop = node_ids[inds_in_pop]
        # then, turn the nodes into a sample set
        sample_set = np.unique(nodes_in_pop)
        sample_sets.append(sample_set)

    # since we most often want many bootstraps, it is much more efficient to
    # calculate the SFS for each window once and then sample from those SFS
    windows = np.linspace(0, ts.sequence_length, num_windows+1)
    windowed_spectra = ts.allele_frequency_spectrum(sample_sets = sample_sets, 
                                                    polarised=True, span_normalise=False,
                                                    windows=windows)
    # then, we can calculate the bootstrap SFS by sampling from the windowed SFS
    bootstrap_spectra = []
    rng = np.random.default_rng(seed)
    for i in range(num_bootstraps):
        window_indices = rng.integers(0, num_windows, size=num_windows)
        bootstrap_sfs = np.zeros(windowed_spectra[0].shape)
        for j in window_indices:
            bootstrap_sfs += windowed_spectra[j]
        bootstrap_sfs = dadi.Spectrum(bootstrap_sfs)
        bootstrap_spectra.append(bootstrap_sfs)
    return bootstrap_spectra


def wrap_model_collapsed(model_func, collapsed_ids, original_ns):
    """
    Wrap a model function to collapse a set of populations.

    Parameters:
    -----------
    model_func : function
        The original demographic model function
    collapsed_ids : list
        Population IDs to collapse
    original_ns : tuple
        Sample sizes from the original (uncollapsed) spectrum.
        This is needed because the underlying model expects the full-dimensional ns.
    """
    def wrapped_model_func(params, ns, pts):
        # Use the original ns for the model (required for proper phi dimensions),
        # then collapse the resulting spectrum
        fs = model_func(params, original_ns, pts)
        fs = fs.combine_pops(collapsed_ids)
        return fs
    wrapped_model_func.__param_names__ = model_func.__param_names__
    return wrapped_model_func

# optimization functions
def opt_single_spectrum(input_file, model_func, initial_params, lower_bounds, upper_bounds,
                        folded=False, collapsed_ids=None,
                        fixed_params=None, num_opts=100, pts_l=[101, 111, 121],
                        output_dir="inference_results", maxeval=600, ll_tol=0.0001,
                        algorithm=nlopt.LN_BOBYQA, func_tol=1e-6, param_tol=1e-6):
    """
    Calculate the SFS from an msprime .trees file
    and optimize parameters for a model function using dadi.
    Note: this function checks for convergence of the log-likelihood and will 
        perform a maximum of num_opts optimizations.    

    Parameters:
    -----------
    input_file : str
        Path to the tree sequence (.trees) file
    model_func : function
        Dadi demographic model function (must have __param_names__ attribute)
    initial_params : list
        Starting values for optimization
    lower_bounds : list
        Lower bounds for parameters
    upper_bounds : list
        Upper bounds for parameters
    folded : bool
        Whether to fold the SFS (default: False)
    collapsed_ids : list
        Population IDs to collapse (None for no collapsing)
    fixed_params : list
        Fixed parameters for optimization (None for unfixed)
    num_opts : int
        Number of optimization runs
    pts_l : list
        Grid points for dadi
    output_dir : str
        Directory to save optimization results
    maxeval : int
        Maximum number of evaluations per optimization
    ll_tol : float
        Tolerance for log-likelihood difference between optimizations (as a percentage; e.g., .1 = 10%)
    algorithm : nlopt algorithm
        Optimization algorithm to use
    func_tol : float
        Absolute tolerance for objective function in optimization
    param_tol : float
        Absolute tolerance for parameters in optimization

    Returns:
    --------
    dict : Status information including success/failure and output file path
    """
    start = time.time()
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the tree sequence
    try:
        ts = tskit.load(input_file)
        sfs = get_sfs_from_ts(ts)
        # we need to define ns from the *original* sfs (the collapsed sfs will cause errors downstream)
        ns = sfs.sample_sizes
    except Exception as e:
        return {
            'success': False,
            'filename': input_file,
            'output_file': None,
            'error': f"Error loading tree sequence: {str(e)}"
        }

    # Fold and collapse if desired
    if collapsed_ids is not None:
        # Wrap model_func first (needs original ns), then collapse the data sfs
        model_func = wrap_model_collapsed(model_func, collapsed_ids, ns)
        sfs = sfs.combine_pops(collapsed_ids)
    if folded:
        sfs = sfs.fold()

    #try:
    extrap_func = dadi.Numerics.make_extrap_func(model_func)

    # Create output file
    base_name = os.path.basename(input_file).replace('.trees', '')
    output_file = os.path.join(output_dir, f"optimization_{base_name}.txt")

    ll_list = []
    i = 0

    if os.path.exists(output_file):
        # Load existing log likelihoods from the file
        with open(output_file, 'r') as f:
            lines = f.readlines()
            # Skip header line, parse log likelihoods from first column
            for line in lines[1:]:
                if line.strip():
                    ll = float(line.strip().split('\t')[0])
                    ll_list.append(np.abs(ll))
                    i += 1
            fid = open(output_file, 'a')
    else:
        # Create a new file and write header
        fid = open(output_file, 'w')
        header = ['log_likelihood'] + model_func.__param_names__ + ['theta']
        fid.write('\t'.join(header) + '\n')
    try:
        while i < num_opts:
            # Check if the best three fits are within ll_tol % of each other
            if i >= 3: # if three or more optimizations completed
                ll_list.sort()
                # here, we really only need to compare the best and third best fits
                if (ll_list[2] - ll_list[0])/ll_list[0] < ll_tol:
                    print(f"Optimization {i} converged within tolerance of {ll_tol}")
                    break
        
            p0 = dadi.Misc.perturb_params(initial_params, fold=1,
                                              upper_bound=upper_bounds,
                                              lower_bound=lower_bounds)

            popt, ll_model = dadi.Inference.opt(p0, sfs, extrap_func, pts_l,
                                                    lower_bound=lower_bounds,
                                                    upper_bound=upper_bounds,
                                                    algorithm=algorithm,
                                                    maxeval=maxeval,
                                                    verbose=10,
                                                    fixed_params=fixed_params, 
                                                    ftol_abs = func_tol, 
                                                    xtol_abs = param_tol)
            ll_list.append(np.abs(ll_model))

            model_fs = extrap_func(popt, ns, pts_l)
            theta0 = dadi.Inference.optimal_sfs_scaling(model_fs, sfs)

            res = [ll_model] + list(popt) + [theta0]
            fid.write('\t'.join([str(ele) for ele in res]) + '\n')
            fid.flush()

            i += 1
            
        fid.close()

        print(f"Finished inference in {time.time() - start} seconds")

        return {
            'success': True,
            'filename': input_file,
            'output_file': output_file,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'filename': input_file,
            'output_file': None,
            'error': str(e)
        }

def get_model_function(ploidyType, model):
    """
    ploidyType: autotetraploid, allotetraploid, autohexaploid, etc.
    model: model name

    Returns:
        model_func: cooresponding dadi demographic model
    """

    if ploidyType == 'autotetraploid':
        import dadi.Polyploidy.auto_demographics as auto_demographics
        import auto_demographic_functions as auto_demographics_supp
        model_map = {
            'snm': auto_demographics.snm,
            'two_epoch': auto_demographics.two_epoch,
            'bottlegrowth': auto_demographics.bottlegrowth,
            'three_epoch': auto_demographics.three_epoch,
            'bottleneck_w_dips': auto_demographics.bottleneck_w_dips,   
            'bottleneck_mig_w_dips': auto_demographics.bottleneck_mig_w_dips,
            'bottleneck_asym_mig_w_dips': auto_demographics.bottleneck_asym_mig_w_dips,
            'bottlegrowth_w_dips': auto_demographics.bottlegrowth_w_dips,
            'bottlegrowth_mig_w_dips': auto_demographics.bottlegrowth_mig_w_dips,
            'bottlegrowth_asym_mig_w_dips': auto_demographics.bottlegrowth_asym_mig_w_dips,
            'triplet_recurrent': auto_demographics_supp.auto_triplet_recurrent,
            'triplet_single_origin': auto_demographics_supp.auto_triplet_single_origin
        }
        return model_map[model]
    elif ploidyType == 'allotetraploid':
        import allo_demographic_functions as allo_demographics
        model_map = {
            'two_epoch': allo_demographics.two_epoch,
            'bottlegrowth': allo_demographics.bottlegrowth,
            'three_epoch': allo_demographics.three_epoch,
            'bottleneck_w_dips': allo_demographics.bottleneck_w_dips,   
            'bottleneck_mig_w_dips': allo_demographics.bottleneck_mig_w_dips,
            'bottleneck_asym_mig_w_dips': allo_demographics.bottleneck_asym_mig_w_dips,
            'bottleneck_noHE_w_dips': allo_demographics.bottleneck_noHE_w_dips,
        }
        return model_map[model]
    else: 
        raise ValueError(f"Unknown ploidyType: {ploidyType}")

def main():
    parser = argparse.ArgumentParser(
        description='Run dadi inference for a single configuration on a single input file'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to input file'
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    configs = config_data['configs']

    for config in configs:
        print(f"Running inference for config: {config['name']}")

        # Required fields
        ploidyType = config['ploidyType']
        model = config['model']
        model_func = get_model_function(ploidyType, model)
        initial_params = config['initial_params']
        lower_bounds = config['lower_bounds']
        upper_bounds = config['upper_bounds']
        output_dir = config['output_dir']

        # Optional fields with defaults
        folded = config.get('folded', False)
        collapsed_ids = config.get('collapsed_ids', None)
        fixed_params = config.get('fixed_params', None)
        num_opts = config.get('num_opts', 100)
        pts_l = config.get('pts_l', [101, 111, 121])
        maxeval = config.get('maxeval', 600)
        algorithm = config.get('algorithm', nlopt.LN_BOBYQA)
        ll_tol = config.get('ll_tol', .0001)
        func_tol = config.get('func_tol', 1e-6)
        param_tol = config.get('param_tol', 1e-6)

        output_dict = opt_single_spectrum(args.input_file, model_func, initial_params, 
                                          lower_bounds, upper_bounds,
                                          folded, collapsed_ids,
                                          fixed_params, num_opts, pts_l,
                                          output_dir, maxeval, ll_tol, algorithm, func_tol, param_tol)

        print(f"Finished optimization for {output_dict['filename']}")
        print(f"Success: {output_dict['success']}")
        print(f"Optimization results saved to {output_dict['output_file']}")
        print(f"Error: {output_dict['error']}")
    
if __name__ == '__main__':
    main()
