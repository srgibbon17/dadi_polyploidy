"""
Define a set of demographic models to exactly mirror the 
msprime models defined in generate_msprime_samples.py.

Example command line usage (for 1D gamma DFE):
python3 $path_to_script/dadi_DFE_inference.py \
    --config-file path_to_config_file/config.yaml \
    --nonsyn-sfs path_to_nonsyn_SFS_file \
    --opt-file path_to_demographic_inference_optimization_results_file \
    --cache1D path_to_1D_cache

Explanation of command line arguments:
    config-file: Path to YAML configuration file (required)
    nonsyn-sfs: Path to nonsynonymous SFS file (output from dadi_demographic_inference.py) (required)
    opt-file: Path to optimization results file from demographic inference (required)
    cache1D: Path to 1D cache file (output from dadi_cache_generation.py) (required)

For a 2D DFE with a mixture model, we would also pass the 2D cache file with the --cache2D argument.

Examples of the config.yaml file for (1) a 1D gamma DFE and (2) a 2D lognormal mixture DFE are provided in the folder.
"""

# note: all of the allotetraploid models considered here include a divergence period 
# between the two diploid populations for T=1 diffusion units. 

import dadi
import dadi.DFE
import numpy as np
import nlopt
import os
import yaml
import argparse
import time
import pickle
import matplotlib.pyplot as plt

# utility functions for getting the selection distribution PDFs from dadi.DFE.PDFs
def get_sel_dist_1d(sel_dist_1d):
    """
    sel_dist_1d: distribution for selection coefficients 
            for 1d or perfectly correlated portion of mixture model

    Returns:
        sel_func_1d: cooresponding dadi DFE PDF
    """
    model_map = {
        'gamma': dadi.DFE.PDFs.gamma,
        'beta': dadi.DFE.PDFs.beta,
        'exponential': dadi.DFE.PDFs.exponential,
        'lognormal': dadi.DFE.PDFs.lognormal,
        'normal': dadi.DFE.PDFs.normal
    }
    
    return model_map[sel_dist_1d]

def get_sel_dist_2d(sel_dist_2d):
    """
    sel_dist_2d: distribution for selection coefficients 
            for 2d or non-perfectly correlated portion of mixture model

    Returns:
        sel_func_2d: cooresponding dadi DFE PDF
    """
    model_map = {
        'gamma': dadi.DFE.PDFs.biv_ind_gamma,
        'lognormal': dadi.DFE.PDFs.biv_lognormal,
    }
    
    return model_map[sel_dist_2d]

# utility function for loading theta_ns from optimization file
def load_theta_from_opt_file(opt_file):
    """
    Load the best theta_syn from the optimization results file.

    Parameters:
    -----------
    opt_file : str
        Path to optimization results file (optimization_*.txt)

    Returns:
    --------
    theta_syn : float
        Best fit theta_syn
    """
    with open(opt_file, 'r') as f:
        lines = f.readlines()
        # Find the row with the highest (least negative) log likelihood
        best_line = None
        best_ll = float('-inf')
        for line in lines[1:]:
            parts = line.strip().split('\t')
            ll = float(parts[0])
            if ll > best_ll:
                best_ll = ll
                best_line = parts
        # get the last parameter (theta_syn)
        theta_syn = float(best_line[-1])
    return theta_syn

# utility function for getting the best fit params for the DFE
def get_best_fit_params(opt_file, dfe_param_names):
    """
    Load the best parameters from the optimization results file.

    Parameters:
    -----------
    opt_file : str
        Path to optimization results file (optimization_*.txt)
    dfe_param_names : list
        Names for DFE parameters (None for default)

    Returns:
    --------
    popt : list
        Best fit parameters (lowest log likelihood)
    """
    with open(opt_file, 'r') as f:
        lines = f.readlines()
        # Find the row with the highest (least negative) log likelihood
        best_line = None
        best_ll = float('-inf')
        for line in lines[1:]:
            parts = line.strip().split('\t')
            ll = float(parts[0])
            if (ll > best_ll) & (ll < 0):
                best_ll = ll
                best_line = parts

        # Extract parameters (skip log_likelihood in first column)
        # subtract one because the model will also include gamma,
        # and thus we would pick up theta unnecessarily
        num_params = len(dfe_param_names)
        popt = [float(best_line[i+1]) for i in range(num_params)]

    return popt

# optimization function
def opt_single_spectrum(input_ns_sfs, theta_ns,
                        DFE_type, sel_dist_1d, sel_dist_2d,
                        cache1D_file, cache2D_file,
                        initial_params, lower_bounds, upper_bounds, fixed_params,
                        misid = False, output_dir="DFE_inference_results", dfe_param_names=None,
                        num_opts=100, maxeval=600, ll_tol=0.0001,
                        algorithm=nlopt.LN_BOBYQA, func_tol=1e-6, param_tol=1e-6):
    """
    Calculate the SFS from an msprime .trees file
    and optimize parameters for a model function using dadi.
    Note: this function checks for convergence of the log-likelihood and will 
        perform a maximum of num_opts optimizations.    

    Parameters:
    -----------
    input_ns_sfs : str
        Path to a nonsynonymous SFS file (output from dadi_demographic_inference.py)
    theta_ns : float
        nonsynonymous theta (=theta_ratio * theta_syn)
    DFE_type : str
        DFE type (1d, 2d, or mixture)
    sel_dist_1d : str
        probability distribution for selection coefficients 
            for 1d or perfectly correlated portion of mixture model
    sel_dist_2d : str
        probability distribution for selection coefficients 
            for 2d or non-perfectly correlated portion of mixture model
    cache1D_file : str
        Path to 1D cache file (output from dadi_cache_generation.py)
    cache2D_file : str
        Path to 2D cache file (output from dadi_cache_generation.py)
    initial_params : list
        Starting values for optimization
    lower_bounds : list
        Lower bounds for parameters
    upper_bounds : list
        Upper bounds for parameters
    fixed_params : list
        Fixed parameters for optimization (None for unfixed)
    misid : bool
        Whether to include ancestral state misidentification (default: False)
    output_dir : str
        Directory to save optimization results
    dfe_param_names : list
        Names for DFE parameters (None for default)
    num_opts : int
        Number of optimization runs
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

    # Load the sfs
    try:
        sfs_ns = dadi.Spectrum.from_file(input_ns_sfs)
    except Exception as e:
        return {
            'success': False,
            'filename': input_ns_sfs,
            'output_file': None,
            'error': f"Error loading site frequency spectrum: {str(e)}"
        }

    # based on DFE type, load the relevant cache(s), 
    #        define dfe_func, and specify func_args
    if DFE_type == '1d':
        cache1d = pickle.load(open(cache1D_file, 'rb'))
        dfe_func = cache1d.integrate
        sel_dist_1d = get_sel_dist_1d(sel_dist_1d)
        func_args = [sel_dist_1d, theta_ns]
    elif DFE_type == '2d':
        cache2d = pickle.load(open(cache2D_file, 'rb'))
        dfe_func = cache2d.integrate
        sel_dist_2d = get_sel_dist_2d(sel_dist_2d)
        func_args = [sel_dist_2d, theta_ns]
    elif DFE_type == 'mixture':
        cache1d = pickle.load(open(cache1D_file, 'rb'))
        cache2d = pickle.load(open(cache2D_file, 'rb'))
        dfe_func = dadi.DFE.mixture
        sel_dist_1d = get_sel_dist_1d(sel_dist_1d)
        sel_dist_2d = get_sel_dist_2d(sel_dist_2d)
        func_args = [cache1d, cache2d, sel_dist_1d, sel_dist_2d, theta_ns]
    else:
        raise ValueError(f"Unknown DFE type: {DFE_type}")


    # add misidentification if desired
    if misid:
        dfe_func = dadi.Numerics.make_anc_state_misid_func(dfe_func)

    # Create output file
    base_name = os.path.basename(input_ns_sfs).replace('.sfs', '')
    output_file = os.path.join(output_dir, f"DFE_optimization_{base_name}.txt")

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
        header = ['log_likelihood'] + dfe_param_names + ['theta']
        fid.write('\t'.join(header) + '\n')
    # run the optimization
    try:
        while i < num_opts:
            # Check if the best three fits are within ll_tol % of each other
            if i >= 10: # if three or more optimizations completed
                ll_list.sort()
                print(ll_list[0], ll_list[1], ll_list[2])
                # here, we really only need to compare the best and third best fits
                if (ll_list[2] - ll_list[0])/ll_list[0] < ll_tol:
                    print(f"Optimization {i} converged within tolerance of {ll_tol}")
                    break
        
            p0 = dadi.Misc.perturb_params(initial_params, fold=1,
                                              upper_bound=upper_bounds,
                                              lower_bound=lower_bounds)

            popt, ll_model = dadi.Inference.opt(p0, sfs_ns, dfe_func, pts=None,
                                                    func_args=func_args,
                                                    lower_bound=lower_bounds,
                                                    upper_bound=upper_bounds,
                                                    algorithm=algorithm,
                                                    maxeval=maxeval,
                                                    multinom=False, # theta_ns is fixed, so multinom=False
                                                    verbose=0,
                                                    fixed_params=fixed_params, 
                                                    ftol_abs = func_tol, 
                                                    xtol_abs = param_tol)
            ll_list.append(np.abs(ll_model))

            # model_fs = dfe_func(popt, sfs_ns.sample_sizes, *func_args, None)

            res = [ll_model] + list(popt) + [theta_ns]
            fid.write('\t'.join([str(ele) for ele in res]) + '\n')
            fid.flush()

            i += 1
            
        fid.close()

        # get the best fit params for the DFE
        p_best = get_best_fit_params(output_file, dfe_param_names)
        print(p_best)

        ns_list = sfs_ns.sample_sizes
        sfs_model = dfe_func(p_best, ns_list, *func_args, None)

        # plot the fit
        fig = plt.figure(123)
        fig.clear()
        if len(ns_list)==1:
            dadi.Plotting.plot_1d_comp_Poisson(sfs_model, sfs_ns, show=False)
        else: # 2d
            dadi.Plotting.plot_2d_comp_Poisson(sfs_model, sfs_ns, vmin=1, show=False)
        fig.savefig(f'''{output_file[:-4]}_plot.pdf''', bbox_inches='tight', format='pdf', dpi=900)

        print(f"Finished inference in {time.time() - start} seconds")

        return {
            'success': True,
            'filename': input_ns_sfs,
            'output_file': output_file,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'filename': input_ns_sfs,
            'output_file': None,
            'error': str(e)
        }

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
        '--nonsyn-sfs',
        type=str,
        required=True,
        help='Path to nonsynonymous SFS file (output from dadi_demographic_inference.py)'
    )
    parser.add_argument(
        '--opt-file',
        type=str,
        required=True,
        help='Path to demographic optimization results file (output from dadi_demographic_inference.py)'
    )
    parser.add_argument(
        '--cache1D',
        type=str,
        default=None,
        help='Path to 1D cache file (output from dadi_cache_generation.py)'
    )
    parser.add_argument(
        '--cache2D',
        type=str,
        default=None,
        help='Path to 2D cache file (output from dadi_cache_generation.py)'
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    configs = config_data['configs']

    for config in configs:
        print(f"Running inference for config: {config['name']}")

        # Required fields
        theta_ratio = config['theta_ratio']
        initial_params = config['initial_params']
        lower_bounds = config['lower_bounds']
        upper_bounds = config['upper_bounds']
        
        # Optional fields with defaults
        DFE_type = config.get('DFE_type', '1d')
        sel_dist_1d = config.get('sel_dist_1d', 'gamma')
        sel_dist_2d = config.get('sel_dist_2d', 'gamma')
        fixed_params = config.get('fixed_params', None)
        misid = config.get('misid', False)
        num_opts = config.get('num_opts', 100)
        maxeval = config.get('maxeval', 600)
        algorithm = config.get('algorithm', nlopt.LN_BOBYQA)
        ll_tol = config.get('ll_tol', .0001)
        func_tol = config.get('func_tol', 1e-6)
        param_tol = config.get('param_tol', 1e-6)
        output_dir = config.get('output_dir', 'DFE_inference_results')
        dfe_param_names = config.get('dfe_param_names', None)

        theta_ns = theta_ratio * load_theta_from_opt_file(args.opt_file)

        output_dict = opt_single_spectrum(args.nonsyn_sfs, theta_ns, 
                                          DFE_type, sel_dist_1d, sel_dist_2d,
                                          args.cache1D, args.cache2D,
                                          initial_params, lower_bounds, upper_bounds, fixed_params,
                                          misid, output_dir, dfe_param_names,
                                          num_opts, maxeval, ll_tol, 
                                          algorithm, func_tol, param_tol)

        print(f"Finished optimization for {output_dict['filename']}")
        print(f"Success: {output_dict['success']}")
        print(f"Optimization results saved to {output_dict['output_file']}")
        print(f"Error: {output_dict['error']}")
    
if __name__ == '__main__':
    main()