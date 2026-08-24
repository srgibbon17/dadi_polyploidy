"""
Example command line usage:
python3 $path_to_script/summarize_inference_results.py \
    --mode summarize \
    --trees-files path_to_tree_sequence_file/*.trees \
    --opt-files path_to_optimization_file/*.txt \
    --output-dir results \
    --output-file nested_summary.csv \
    --num-windows 33 \
    --num-bootstraps 100 \
    --seed 42 \
    --nested \
    --config $base_path/summary_configs_ex.yaml

Mode: summarize (GIM uncertainties), lrt (likelihood ratio test), 
      claic_clbic (compute CLAIC and CLBIC), or best_fits (extract best fits)
Number of windows: number of windows to split the genome into for computing boostrap spectra
                   This can be calculated by looking at the LD decay for the given demographic model
Number of bootstraps: number of bootstrap replicate spectra to compute
Nested flag: whether to use nested_model from config for GIM uncertainties instead of model
Log flag: whether to evaluate H, J, and GIM in log space for uncertainties
Config: path to YAML config file with ploidyType, model, pts_l, folded, collapsed_ids, eps_l, nested_indices, weights, true_params
        See summary_configs_ex.yaml for an example and explanation of each parameter
"""

import os
import numpy as np
import pandas as pd
import tskit
import dadi
import dadi.Godambe

# utility functions for cleaning up optimization files
# (for some odd reason, a few do not have a header)
def add_header_to_optimization_file(filepath, model_func):
    """
    Add a header to an existing optimization file that is missing one.

    Parameters:
    -----------
    filepath : str
        Path to the optimization results file
    model_func : function
        Dadi demographic model function (must have __param_names__ attribute)

    Returns:
    --------
    bool : True if header was added, False if header already exists
    """
    # Read existing content
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        # Empty file, just write header
        header = ['log_likelihood'] + model_func.__param_names__ + ['theta']
        with open(filepath, 'w') as f:
            f.write('\t'.join(header) + '\n')
        return True

    # Check if first line is already a header (contains 'log_likelihood')
    first_line = lines[0].strip()
    if first_line.startswith('log_likelihood'):
        return False

    # Prepend header to existing content
    header = ['log_likelihood'] + model_func.__param_names__ + ['theta']
    with open(filepath, 'w') as f:
        f.write('\t'.join(header) + '\n')
        f.writelines(lines)

    return True

# utility function for mapping model names to model functions
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
            'two_epoch_noHE': allo_demographics.two_epoch_noHE,
            'bottlegrowth': allo_demographics.bottlegrowth,
            'bottlegrowth_noHE': allo_demographics.bottlegrowth_noHE,
            'three_epoch': allo_demographics.three_epoch,
            'three_epoch_noHE': allo_demographics.three_epoch_noHE,
            'bottleneck_w_dips': allo_demographics.bottleneck_w_dips,   
            'bottleneck_mig_w_dips': allo_demographics.bottleneck_mig_w_dips,
            'bottleneck_asym_mig_w_dips': allo_demographics.bottleneck_asym_mig_w_dips,
            'bottleneck_noHE_w_dips': allo_demographics.bottleneck_noHE_w_dips,
        }
        return model_map[model]
    else: 
        raise ValueError(f"Unknown ploidyType: {ploidyType}")

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

# utility function for extracting best fit from optimization file
def extract_best_fit(opt_file):
    """
    Extract the best fit (highest log-likelihood) from an optimization file.

    Parameters:
    -----------
    opt_file : str
        Path to optimization .txt file

    Returns:
    --------
    dict : Best fit parameters including log_likelihood, all params, and theta
    """
    df = pd.read_csv(opt_file, sep='\t')
    best_idx = df['log_likelihood'].idxmax()
    best_fit = df.loc[best_idx].to_dict()
    best_fit['optimization_file'] = opt_file
    return best_fit

# utility function for wrapping model functions to collapse populations
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

# utility function for computing uncertainties using GIM
def compute_gim_uncertainties(trees_file, model_func, best_params, pts_l,
                               num_windows=100, num_bootstraps=100, seed=42, eps_l=[0.01],
                               folded=False, collapsed_ids=None, log=False):
    """
    Compute parameter uncertainties using the Godambe Information Matrix.

    Parameters:
    -----------
    trees_file : str
        Path to the .trees file
    model_func : function
        dadi demographic model function
    best_params : list
        Best-fit parameter values from optimization
    pts_l : list
        Grid points for extrapolation
    num_windows : int
        Number of windows for bootstrap
    num_bootstraps : int
        Number of bootstrap replicates
    seed : int
        Random seed for reproducibility (for making bootstrap SFS)
    eps_l : list
        List of epsilon tolerances for numerical derivative step size
    folded : bool
        Whether to fold the SFS
    collapsed_ids : list
        Population IDs to collapse
    log : bool
        Whether to evaluate H, J, and GIM in log space

    Returns:
    --------
    Uncertainties : list
        List of lists of uncertainties for each parameter (including theta if multinom=True)
        One list for each value of eps_l
    """
    # Load tree sequence and generate data SFS
    ts = tskit.load(trees_file)
    sfs = get_sfs_from_ts(ts)

    ns = sfs.sample_sizes

    # Wrap the model function to collapse and fold if needed
    if collapsed_ids is not None:
        model_func = wrap_model_collapsed(model_func, collapsed_ids, ns)
        sfs = sfs.combine_pops(collapsed_ids)
    if folded:
        sfs = sfs.fold()

    # Generate bootstrap spectra
    bootstrap_spectra = get_bootstrap_spectra(ts, num_windows, num_bootstraps, seed)

    # Apply folding/collapsing to bootstraps if needed
    if collapsed_ids is not None or folded:
        processed_bootstraps = []
        for fs in bootstrap_spectra:
            if collapsed_ids is not None:
                fs = fs.combine_pops(collapsed_ids)
            if folded:
                fs = fs.fold()
            processed_bootstraps.append(fs)
        bootstrap_spectra = processed_bootstraps

    # Create extrapolation function
    extrap_func = dadi.Numerics.make_extrap_func(model_func)
    uncerts_list = []

    # Compute GIM uncertainties
    # dadi.Godambe.GIM_uncert returns (uncerts, GIM)
    for eps in eps_l:
        uncerts = dadi.Godambe.GIM_uncert(
            extrap_func, pts_l, bootstrap_spectra, best_params, sfs,
            multinom=True, eps=eps, log=log
        )
        uncerts_list.append(uncerts)

    return uncerts_list

# utility function for computing CLAIC and CLBIC
def compute_CLAIC_CLBIC(trees_file, model_func, best_params, pts_l, ll,
                               num_windows=100, num_bootstraps=100, seed=42, 
                               eps_l=[0.01], folded=False, collapsed_ids=None):
    """
    Compute CLAIC and CLBIC given the effective number of parameters tr(H*G^-1).

    Parameters:
    -----------
    trees_file : str
        Path to the .trees file
    model_func : function
        dadi demographic model function
    best_params : list
        Best-fit parameter values from optimization
    pts_l : list
        Grid points for extrapolation
    ll : float
        Log-likelihood of the data
    num_windows : int
        Number of windows for bootstrap
    num_bootstraps : int
        Number of bootstrap replicates
    seed : int
        Random seed for reproducibility (for making bootstrap SFS)
    eps_l : list
        List of epsilon tolerances for numerical derivative step size
    folded : bool
        Whether to fold the SFS
    collapsed_ids : list
        Population IDs to collapse

    Returns:
    --------
    CLAIC_list : list
        List of CLAIC values (of length eps_l)
    CLBIC_list : list
        List of CLBIC values (of length eps_l)
    """
    # Load tree sequence and generate data SFS
    ts = tskit.load(trees_file)
    sfs = get_sfs_from_ts(ts)

    ns = sfs.sample_sizes

    # Wrap the model function to collapse and fold if needed
    if collapsed_ids is not None:
        model_func = wrap_model_collapsed(model_func, collapsed_ids, ns)
        sfs = sfs.combine_pops(collapsed_ids)
    if folded:
        sfs = sfs.fold()

    # Generate bootstrap spectra
    bootstrap_spectra = get_bootstrap_spectra(ts, num_windows, num_bootstraps, seed)

    # Apply folding/collapsing to bootstraps if needed
    if collapsed_ids is not None or folded:
        processed_bootstraps = []
        for fs in bootstrap_spectra:
            if collapsed_ids is not None:
                fs = fs.combine_pops(collapsed_ids)
            if folded:
                fs = fs.fold()
            processed_bootstraps.append(fs)
        bootstrap_spectra = processed_bootstraps

    # Create extrapolation function
    extrap_func = dadi.Numerics.make_extrap_func(model_func)
    CLAIC_list = []
    CLBIC_list = []

    # Compute GIM uncertainties
    # dadi.Godambe.GIM_uncert returns (uncerts, GIM)
    for eps in eps_l:
        effective_dim = dadi.Godambe.effective_dimension(
            extrap_func, pts_l, bootstrap_spectra, best_params, sfs,
            multinom=True, eps=eps)
        CLAIC = -2*ll + 2*effective_dim
        CLBIC = -2*ll + np.log(sfs.sum())*effective_dim
        CLAIC_list.append(CLAIC)
        CLBIC_list.append(CLBIC)

    return CLAIC_list, CLBIC_list

# utility function for computing adjusted LRT for pairs of inferred parameters
def compute_adjusted_lrt(nested_ll, full_ll, params_nested, nested_indices,
                         trees_file, full_model_func, pts_l, 
                         num_windows=100, num_bootstraps=100, seed=42,
                         folded=False, collapsed_ids=None, eps_l=[0.01]):
    """
    Compute the adjusted likelihood ratio test statistic using Godambe.

    Parameters:
    -----------
    nested_ll : float
        Log-likelihood of nested model
    full_ll : float
        Log-likelihood of full model
    params_nested : list
        Best-fit parameters for nested model 
        (= length as # params for full/complex model)
    nested_indices : list
        Indices of nested parameters in complex model
    trees_file : str
        Path to the .trees file
    full_model_func : function
        dadi demographic model function for full/complex model
    pts_l : list
        Grid points for extrapolation
    num_windows : int
        Number of windows for bootstrap
    num_bootstraps : int
        Number of bootstrap replicates
    seed : int
        Random seed
    folded : bool
        Whether to fold the SFS
    collapsed_ids : list
        Population IDs to collapse
    eps_l : list
        List of epsilon tolerance for numerical derivative step size

    Returns:
    --------
    adj_lrt : float
        List of adjusted LRT statistics (one for each eps_l)
    """
    # Load tree sequence and generate data SFS
    ts = tskit.load(trees_file)
    sfs = get_sfs_from_ts(ts)
    ns = sfs.sample_sizes

    if collapsed_ids is not None:
        full_model_func = wrap_model_collapsed(full_model_func, collapsed_ids, ns)
        sfs = sfs.combine_pops(collapsed_ids)
    if folded:
        sfs = sfs.fold()


    # Generate bootstrap spectra
    bootstrap_spectra = get_bootstrap_spectra(ts, num_windows, num_bootstraps, seed)

    # Apply folding/collapsing to bootstraps if needed
    if collapsed_ids is not None or folded:
        processed_bootstraps = []
        for fs in bootstrap_spectra:
            if collapsed_ids is not None:
                fs = fs.combine_pops(collapsed_ids)
            if folded:
                fs = fs.fold()
            processed_bootstraps.append(fs)
        bootstrap_spectra = processed_bootstraps

    # Create extrapolation functions
    full_extrap = dadi.Numerics.make_extrap_func(full_model_func)

    adj_lrts = []

    # Compute adjusted LRT
    # dadi.Godambe.LRT_adjust returns (D_adj, p_value)
    for eps in eps_l:
        adj = dadi.Godambe.LRT_adjust(
            full_extrap, pts_l, bootstrap_spectra, params_nested, sfs,
            nested_indices, multinom=True, eps=eps
        )
    
        adj_lrt = adj*2*(full_ll - nested_ll)
        adj_lrts.append(adj_lrt)

    return adj_lrts

def add_headers(opt_files, config):
    """
    Add headers to optimization files that are missing one.

    Parameters:
    -----------
    opt_files : list of str
        List of optimization .txt files
    config : dict
        Configuration dict with keys: ploidyType, model, pts_l, folded,
        collapsed_ids, fixed_params

    Returns:
    --------
    None
    """
    model_func = get_model_function(ploidyType = config['ploidyType'], 
                                    model = config['model'])
    
    for opt_file in opt_files:
        add_header_to_optimization_file(opt_file, model_func)

    return

def summarize_with_uncertainties(
    opt_files,
    trees_files,
    config,
    output_dir="inference_results",
    output_file="inference_summary.csv",
    num_windows=100,
    num_bootstraps=100,
    seed=42,
    nested=False,
    log=False
):
    """
    Summarize inference results with GIM uncertainties.

    Parameters:
    -----------
    opt_files : list
        List of optimization .txt file paths
    trees_files : list
        List of corresponding .trees file paths (parallel to opt_files)
    config : dict
        Configuration dict with keys: ploidyType, model, pts_l, folded,
        collapsed_ids. If nested=True, must also include nested_model.
    output_dir : str
        Directory to save output
    output_file : str
        Name of output file
    num_windows : int
        Number of windows for bootstrap
    num_bootstraps : int
        Number of bootstrap replicates
    seed : int
        Random seed
    nested : bool
        If True, use nested_model from config for GIM uncertainties instead of model
    log : bool
        Whether to evaluate H, J, and GIM in log space

    Returns:
    --------
    pd.DataFrame : Summary with best fits and uncertainties
    """
    if len(opt_files) != len(trees_files):
        raise ValueError("opt_files and trees_files must have the same length")

    # add headers
    add_headers(opt_files, config)

    # sort the input data by file name
    opt_files.sort()
    trees_files.sort()

    os.makedirs(output_dir, exist_ok=True)

    # Get model function for reading parameters from optimization files
    model_func = get_model_function(config['ploidyType'], config['model'])

    # Get model function for GIM uncertainties (may be nested model)
    if nested:
        if 'nested_model' not in config:
            raise ValueError("nested=True but 'nested_model' not found in config")
        gim_model_func = get_model_function(config['ploidyType'], config['nested_model'])
        print(f"Using nested model for GIM: {config['nested_model']}")
    else:
        gim_model_func = model_func

    pts_l = config.get('pts_l', [101, 111, 121])
    folded = config.get('folded', False)
    collapsed_ids = config.get('collapsed_ids', None)
    eps_l = config.get('eps_l', [0.1, 0.01, 0.001])
    true_params = config.get('true_params', None)

    results = []

    for opt_file, trees_file in zip(opt_files, trees_files):
        print(f"Processing: {opt_file}")

        # Extract best fit
        best_fit = extract_best_fit(opt_file)

        # Get parameter names from the optimization file's model (full model)
        param_names = model_func.__param_names__
        best_params = [best_fit[p] for p in param_names]

        # Get parameter names for GIM model (may differ if nested)
        # Only extract parameters that exist in the GIM model
        gim_param_names = gim_model_func.__param_names__
        gim_best_params = []
        for p in gim_param_names:
            if p in best_fit:
                gim_best_params.append(best_fit[p])
            else:
                raise ValueError(f"Parameter '{p}' required by GIM model not found in optimization file")

        # Compute uncertainties using GIM model
        gim_uncerts = compute_gim_uncertainties(
                trees_file, gim_model_func, gim_best_params, pts_l,
                num_windows, num_bootstraps, seed, eps_l,
                folded, collapsed_ids, log
            )

        # add theta to the param_names for output
        param_names_extended = list(param_names) + ['theta']
        all_params = best_params + [best_fit['theta']]

        # GIM param names extended (for mapping uncertainties)
        gim_param_names_extended = list(gim_param_names) + ['theta']

        # loop through eps_l and add results to row
        for i, eps in enumerate(eps_l):
            uncerts = gim_uncerts[i]

            # Build result row
            row = {
            'trees_file': trees_file,
            'optimization_file': opt_file,
            'log_likelihood': best_fit['log_likelihood'],
            'theta': best_fit['theta'],
            'eps': eps
            }

            # Add parameters and uncertainties
            # Map uncertainties from GIM model parameters to full model parameters
            for j, pname in enumerate(param_names_extended):
                row[pname] = all_params[j]
                # Check if this parameter has an uncertainty from GIM
                if pname in gim_param_names_extended:
                    gim_idx = gim_param_names_extended.index(pname)
                    row[f'{pname}_uncert'] = uncerts[gim_idx] if gim_idx < len(uncerts) else np.nan
                else:
                    row[f'{pname}_uncert'] = np.nan  # Parameter not in GIM model
                row[f'{pname}_true'] = true_params[j] if true_params and j < len(true_params) else np.nan

            results.append(row)

    summary_df = pd.DataFrame(results)

    # Save summary
    summary_file = os.path.join(output_dir, output_file)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    return summary_df


def summarize_best_fits(
    opt_files,
    trees_files,
    config,
    output_dir="inference_results",
    output_file="best_fits.csv"
):
    """
    Summarize inference results with GIM uncertainties.

    Parameters:
    -----------
    opt_files : list
        List of optimization .txt file paths
    trees_files : list
        List of corresponding .trees file paths (parallel to opt_files)
    config : dict
        Configuration dict with keys: ploidyType, model, pts_l, folded,
        collapsed_ids. If nested=True, must also include nested_model.
    output_dir : str
        Directory to save output
    output_file : str
        Name of output file
    num_windows : int
        Number of windows for bootstrap
    num_bootstraps : int
        Number of bootstrap replicates
    seed : int
        Random seed
    nested : bool
        If True, use nested_model from config for GIM uncertainties instead of model
    log : bool
        Whether to evaluate H, J, and GIM in log space

    Returns:
    --------
    pd.DataFrame : Summary with best fits and uncertainties
    """
    if len(opt_files) != len(trees_files):
        raise ValueError("opt_files and trees_files must have the same length")

    # add headers
    add_headers(opt_files, config)

    # sort the input data by file name
    opt_files.sort()
    trees_files.sort()

    os.makedirs(output_dir, exist_ok=True)

    # Get model function for reading parameters from optimization files
    model_func = get_model_function(config['ploidyType'], config['model'])

    true_params = config.get('true_params', None)

    results = []

    for opt_file, trees_file in zip(opt_files, trees_files):
        print(f"Processing: {opt_file}")

        # Extract best fit
        best_fit = extract_best_fit(opt_file)

        # Get parameter names from the optimization file's model (full model)
        param_names = model_func.__param_names__
        best_params = [best_fit[p] for p in param_names]

        # add theta to the param_names for output
        param_names_extended = list(param_names) + ['theta']
        all_params = best_params + [best_fit['theta']]

        # Build result row
        row = {
        'trees_file': trees_file,
        'optimization_file': opt_file,
        'log_likelihood': best_fit['log_likelihood'],
        'theta': best_fit['theta']
        }

        # Add parameters and uncertainties
        # Map uncertainties from GIM model parameters to full model parameters
        for j, pname in enumerate(param_names_extended):
            row[pname] = all_params[j]
            row[f'{pname}_true'] = true_params[j] if true_params and j < len(true_params) else np.nan

        results.append(row)

    summary_df = pd.DataFrame(results)

    # Save summary
    summary_file = os.path.join(output_dir, output_file)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    return summary_df

def summarize_with_CLAIC_CLBIC(
    opt_files,
    trees_files,
    config,
    output_dir="IC_results",
    output_file="IC_summary.csv",
    num_windows=100,
    num_bootstraps=100,
    seed=42,
    nested=False
):
    """
    Summarize inference results with GIM uncertainties.

    Parameters:
    -----------
    opt_files : list
        List of optimization .txt file paths
    trees_files : list
        List of corresponding .trees file paths (parallel to opt_files)
    config : dict
        Configuration dict with keys: ploidyType, model, pts_l, folded,
        collapsed_ids. If nested=True, must also include nested_model.
    output_dir : str
        Directory to save output
    output_file : str
        Name of output file
    num_windows : int
        Number of windows for bootstrap
    num_bootstraps : int
        Number of bootstrap replicates
    seed : int
        Random seed
    nested : bool
        If True, use nested_model from config for GIM uncertainties instead of model

    Returns:
    --------
    pd.DataFrame : Summary with best fits and uncertainties
    """
    if len(opt_files) != len(trees_files):
        raise ValueError("opt_files and trees_files must have the same length")

    # add headers if they are not already there
    add_headers(opt_files, config)

    # sort the input data by file name
    opt_files.sort()
    trees_files.sort()

    os.makedirs(output_dir, exist_ok=True)

    # Get model function for reading parameters from optimization files
    model_func = get_model_function(config['ploidyType'], config['model'])

    # Get model function for GIM uncertainties (may be nested model)
    if nested:
        if 'nested_model' not in config:
            raise ValueError("nested=True but 'nested_model' not found in config")
        gim_model_func = get_model_function(config['ploidyType'], config['nested_model'])
        print(f"Using nested model for GIM: {config['nested_model']}")
    else:
        gim_model_func = model_func

    pts_l = config.get('pts_l', [101, 111, 121])
    folded = config.get('folded', False)
    collapsed_ids = config.get('collapsed_ids', None)
    eps_l = config.get('eps_l', [0.1, 0.01, 0.001])

    results = []

    for opt_file, trees_file in zip(opt_files, trees_files):
        print(f"Processing: {opt_file}")

        # Extract best fit
        best_fit = extract_best_fit(opt_file)

        ll = best_fit['log_likelihood']

        # Get parameter names for GIM model (may differ if nested)
        # Only extract parameters that exist in the GIM model
        gim_param_names = gim_model_func.__param_names__
        gim_best_params = []
        for p in gim_param_names:
            if p in best_fit:
                gim_best_params.append(best_fit[p])
            else:
                raise ValueError(f"Parameter '{p}' required by GIM model not found in optimization file")

        # Compute CLAIC/CLBIC using GIM model
        CLAIC_l, CLBIC_l = compute_CLAIC_CLBIC(
                trees_file, gim_model_func, gim_best_params, pts_l,
                ll, num_windows, num_bootstraps, seed, eps_l,
                folded, collapsed_ids
            )
        
        ts = tskit.load(trees_file)
        sfs = get_sfs_from_ts(ts)
        # compute AIC and BIC too
        AIC = -2*ll + 2*len(gim_param_names)
        BIC = -2*ll + np.log(sfs.sum())*len(gim_param_names)
        
         # loop through eps_l and add results to row
        for i, eps in enumerate(eps_l):
            CLAIC = CLAIC_l[i]
            CLBIC = CLBIC_l[i]

            row = {
                'trees_file': trees_file,
                'll': ll,
                'AIC': AIC,
                'CLAIC': CLAIC,
                'BIC': BIC,
                'CLBIC': CLBIC,
                'eps': eps
            }

            results.append(row)

    IC_df = pd.DataFrame(results)

    # Save LRT results
    IC_file = os.path.join(output_dir, output_file)
    IC_df.to_csv(IC_file, index=False)
    print(f"LRT results saved to: {IC_file}")

    return IC_df

def compute_lrt_for_pairs(
    nested_opt_files,
    full_opt_files,
    trees_files,
    config,
    output_dir="inference_results",
    output_file="lrt_results.csv",
    num_windows=100,
    num_bootstraps=100,
    seed=42
):
    """
    Compute adjusted LRT for pairs of nested and full model optimizations.

    Parameters:
    -----------
    nested_opt_files : list
        List of optimization .txt files for nested model
    full_opt_files : list
        List of optimization .txt files for full model
    trees_files : list
        List of corresponding .trees file paths
    config : dict
        Configuration for inference
    output_dir : str
        Directory to save output
    output_file : str
        Name of output file
    num_windows : int
        Number of windows for bootstrap
    num_bootstraps : int
        Number of bootstrap replicates
    seed : int
        Random seed

    Returns:
    --------
    pd.DataFrame : LRT results for each pair
    """
    if not (len(nested_opt_files) == len(full_opt_files) == len(trees_files)):
        raise ValueError("All input lists must have the same length")
    
    # add headers
    add_headers(nested_opt_files, config)
    add_headers(full_opt_files, config)

    # sort the input data by file name
    nested_opt_files.sort()
    full_opt_files.sort()
    trees_files.sort()

    os.makedirs(output_dir, exist_ok=True)

    # Get model functions
    model = get_model_function(config['ploidyType'], config['model'])

    pts_l = config.get('pts_l', [101, 111, 121])
    folded = config.get('folded', False)
    collapsed_ids = config.get('collapsed_ids', None)
    nested_indices = config.get('nested_indices', None) 
    eps_l = config.get('eps_l', [0.1, 0.01, 0.001]) 
    weights = config.get('weights', [0.5, 0.5]) # weights for chi-sq statistic

    param_names = model.__param_names__

    results = []

    for nested_opt, full_opt, trees_file in zip(nested_opt_files, full_opt_files, trees_files):
        print(f"Computing LRT for: {trees_file}")

        # Extract best fits
        nested_fit = extract_best_fit(nested_opt)
        full_fit = extract_best_fit(full_opt)

        nested_ll = nested_fit['log_likelihood']
        full_ll = full_fit['log_likelihood']

        nested_params = [nested_fit[p] for p in param_names]

        # Compute adjusted LRT
        #try:
        D_adj_l = compute_adjusted_lrt(
                nested_ll, full_ll,
                nested_params, nested_indices,
                trees_file, model,
                pts_l, num_windows, num_bootstraps, seed,
                folded, collapsed_ids, eps_l
            )
        #except Exception as e:
        #    print(f"  Warning: LRT computation failed: {e}")
        #    D_adj_l = [np.nan] * len(eps_l)

        # loop through eps_l and add results to row
        for i, eps in enumerate(eps_l):
            D_adj = D_adj_l[i]

            row = {
                'trees_file': trees_file,
                'nested_opt_file': nested_opt,
                'full_opt_file': full_opt,
                'nested_ll': nested_ll,
                'full_ll': full_ll,
                'D_unadj': 2*(full_ll - nested_ll),
                'D_adj': D_adj,
                'p_value_unadj': dadi.Godambe.sum_chi2_ppf(2*(full_ll - nested_ll), weights),
                'p_value_adj': dadi.Godambe.sum_chi2_ppf(D_adj, weights),
                'eps': eps
            }

            results.append(row)

    lrt_df = pd.DataFrame(results)

    # Save LRT results
    lrt_file = os.path.join(output_dir, output_file)
    lrt_df.to_csv(lrt_file, index=False)
    print(f"LRT results saved to: {lrt_file}")

    return lrt_df


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Summarize dadi inference results with GIM uncertainties or compute LRT'
    )

    # Mode
    parser.add_argument('--mode', choices=['summarize', 'lrt', 'claic_clbic', 'best_fits'], required=True,
                        help='Analysis mode: summarize (GIM uncertainties), lrt (likelihood ratio test), ' \
                        'claic_clbic (compute CLAIC and CLBIC), or best_fits (extract best fits)')

    # Input files
    parser.add_argument('--trees-files', nargs='+', required=True,
                        help='Tree sequence files (.trees)')
    parser.add_argument('--opt-files', nargs='+', required=True,
                        help='Optimization result files (.txt)')
    parser.add_argument('--opt-files-2', nargs='+', default=None,
                        help='Second set of optimization files (required for LRT mode - full model files)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Output directory')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output filename (default: inference_summary.csv or lrt_results.csv)')

    # Bootstrap options
    parser.add_argument('--num-windows', type=int, default=100,
                        help='Number of windows for bootstrap')
    parser.add_argument('--num-bootstraps', type=int, default=100,
                        help='Number of bootstrap replicates')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # GIM option
    parser.add_argument('--nested', action='store_true', default=False,
                        help='Use nested_model from config for GIM uncertainties instead of model')
    parser.add_argument('--log', action='store_true', default=False,
                        help='Evaluate H, J, and GIM in log space for uncertainties')

    # Config as YAML file
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file with: ploidyType, model, pts_l, folded, '
                             'collapsed_ids, eps_l, nested_indices, weights, true_params. '
                             'If --nested is used, must also include nested_model.')

    args = parser.parse_args()

    # Parse YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set default output filename based on mode
    if args.output_file is None:
        args.output_file = 'inference_summary.csv' if args.mode == 'summarize' else 'lrt_results.csv'

    if args.mode == 'summarize':
        summarize_with_uncertainties(
            opt_files=args.opt_files,
            trees_files=args.trees_files,
            config=config,
            output_dir=args.output_dir,
            output_file=args.output_file,
            num_windows=args.num_windows,
            num_bootstraps=args.num_bootstraps,
            seed=args.seed,
            nested=args.nested,
            log=args.log
        )

    elif args.mode == 'claic_clbic':
        summarize_with_CLAIC_CLBIC(
            opt_files=args.opt_files,
            trees_files=args.trees_files,
            config=config,
            output_dir=args.output_dir,
            output_file=args.output_file,
            num_windows=args.num_windows,
            num_bootstraps=args.num_bootstraps,
            seed=args.seed,
            nested=args.nested
        )

    elif args.mode == 'lrt':
        if args.opt_files_2 is None:
            parser.error("--opt-files-2 is required for LRT mode (full model optimization files)")

        compute_lrt_for_pairs(
            nested_opt_files=args.opt_files,
            full_opt_files=args.opt_files_2,
            trees_files=args.trees_files,
            config=config,
            output_dir=args.output_dir,
            output_file=args.output_file,
            num_windows=args.num_windows,
            num_bootstraps=args.num_bootstraps,
            seed=args.seed
        )

    elif args.mode == 'best_fits':
        
        summarize_best_fits(
            opt_files=args.opt_files,
            trees_files=args.trees_files,
            config=config,
            output_dir=args.output_dir,
            output_file=args.output_file
        )
