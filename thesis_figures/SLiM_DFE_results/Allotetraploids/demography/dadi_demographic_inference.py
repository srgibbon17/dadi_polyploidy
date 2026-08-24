"""
Define a set of demographic models to exactly mirror the 
msprime models defined in generate_msprime_samples.py.
"""

# note: all of the allotetraploid models considered here include a divergence period 
# between the two diploid populations for T=1 diffusion units. 

import dadi
import dadi.Polyploidy.Integration as PolyInt
import numpy as np
import nlopt
import os
import tskit, msprime
import yaml
import argparse
import time
import re

# utility function for getting Q from a tree sequence filename
def get_Q_from_filepath(filepath):
    """
    Extract Q value from a tree sequence filename.
    Example: 'auto_final_Q_5_REP_1_ID_1.trees' -> 5
    """
    basename = os.path.basename(filepath)
    match = re.search(r'Q_(\d+)', basename)
    if not match:
        raise ValueError(f"Could not extract Q number from filename: {basename}")
    return int(match.group(1))

# utility function for getting Q from a tree sequence filename
def get_rep_from_filepath(filepath):
    """
    Extract rep value from a tree sequence filename.
    Example: 'auto_final_Q_5_REP_1_ID_1.trees' -> 5
    """
    basename = os.path.basename(filepath)
    match = re.search(r'REP_(\d+)', basename)
    if not match:
        raise ValueError(f"Could not extract rep number from filename: {basename}")
    return int(match.group(1))

# utility functions for calculating spectra from tree sequences

# note: the below functions are written to work with SLiM simulations, so it also 
# (1) overlays neutral mutations onto the tree sequence and writes this mutated ts
# and (2) samples from the tree sequence

# this is in opposition to the simulations from msprime which already have the neutral mutations
# and only contain the sampled individuals

def get_sample_sets(pop_ids, node_ids, ns_list_copy, paired_pops, rng):
    """
    Build per-population lists of sampled individuals' node ids.

    pop_ids: ts.individuals_population array (population id of each individual)
    node_ids: ts.individuals_nodes array (node ids of each individual, ragged)
    ns_list_copy: number of individuals to sample per population, in ascending
        population-id order (i.e. aligned with np.unique(pop_ids))
    paired_pops: list of lists of population ids whose individuals must be sampled
        by matching rank rather than independently, e.g. [[2, 3]] for an autotetraploid
        whose two subgenomes are recorded as separate diploid populations 2 and 3, where
        the individual at a given rank within population 2 and the individual at the same
        rank within population 3 share the same original (SLiM-tagged) individual.
        Populations not listed in any pair are sampled independently. None disables pairing.
    rng: np.random.Generator used for sampling

    Returns:
    samples_list: list of arrays of node ids, one entry per population, ordered to
        match np.unique(pop_ids)
    """
    unique_pops = np.unique(pop_ids)
    if len(ns_list_copy) != len(unique_pops):
        raise ValueError(
            f"ns_list has {len(ns_list_copy)} entries but the tree sequence has "
            f"{len(unique_pops)} populations with individuals: {list(unique_pops)}"
        )
    ns_by_pop = dict(zip(unique_pops, ns_list_copy))

    # default: every population is its own (unpaired) group
    pop_to_group = {pop: [pop] for pop in unique_pops}
    for group in (paired_pops or []):
        missing = [p for p in group if p not in ns_by_pop]
        if missing:
            raise ValueError(f"paired_pops references population(s) {missing} not "
                              f"present in the tree sequence (found {list(unique_pops)})")
        for pop in group:
            pop_to_group[pop] = list(group)

    sampled_nodes_by_pop = {}
    for pop in unique_pops:
        if pop in sampled_nodes_by_pop:
            continue
        group = pop_to_group[pop]

        # individuals in each population, already ascending by individual id
        inds_by_pop = {p: np.where(pop_ids == p)[0] for p in group}

        ns_values = {ns_by_pop[p] for p in group}
        if len(ns_values) != 1:
            raise ValueError(f"Paired populations {group} must request the same sample "
                              f"size, got {[ns_by_pop[p] for p in group]}")
        ns = ns_values.pop()

        counts = {p: len(inds_by_pop[p]) for p in group}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Paired populations {group} do not have matching numbers "
                              f"of individuals to pair by rank: {counts}")

        # sample the same ranks across every population in the group, so the
        # individual at a given rank in each population is treated as the same
        # underlying (SLiM-tagged) individual
        rank_idx = rng.choice(counts[group[0]], size=ns, replace=False)
        for p in group:
            sampled_inds = inds_by_pop[p][rank_idx]
            sampled_nodes_by_pop[p] = np.unique(node_ids[sampled_inds])

    return [sampled_nodes_by_pop[pop] for pop in unique_pops]

def get_sfs_from_ts(ts_path, ns_list, n_reps, Q=5, mut_seed=123, sampling_seed=42, paired_pops=None):
    """
    Function to compute the SFS from an arbitrary tree sequence.

    ts_path: path to a single tskit tree sequence object
    ns_list: a list of sample sizes (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 4 haplotypes is equivalent to sampling 1 autotetraploid
    n_reps: number of replicates to load from the ts_path
    Q: scaling factor from the SLiM simulation
    mut_seed: seed for the random number generator for overlaying neutral mutations
    sampling_seed: seed for the random number generator for sampling individuals
    paired_pops: list of lists of population ids whose individuals must be sampled together
        (e.g. an autotetraploid split across populations 2 and 3 as separate diploid
        "subgenome" individuals sharing the same underlying SLiM tag). Pass [[2, 3]] to draw
        matching individuals from populations 2 and 3; populations not listed are sampled
        independently. None (default) samples every population independently.

    Returns:
    sfs_syn, sfs_ns: dadi Spectrum objects synonymous and nonsynonymous mutations
    """
    # set up a reproducible rng for overlaying neutral mutations
    rep_id = get_rep_from_filepath(ts_path)
    mut_rng = np.random.default_rng(mut_seed + 100*rep_id)
    mut_seeds = mut_rng.integers(1, 2**32, size=n_reps)

    # convert the sizes of the sfs to be in number of individuals
    ns_list_copy = [int(ns/2) for ns in ns_list]
    if any(ns % 2 != 0 for ns in ns_list):
        raise ValueError("ns must be a multiple of 2.")

    # get the base path from the ts_path
    base_path = re.sub(r'_ID_\d+\.trees$', '', ts_path)
    # use rep 1 to establish sample sets
    ts0 = tskit.load(f"{base_path}_ID_1.trees")
    # extract the nodes and populations from the ts
    node_ids = ts0.individuals_nodes
    pop_ids = ts0.individuals_population
    # set up a reproducible rng for sampling
    sample_rng = np.random.default_rng(sampling_seed+rep_id*100)
    # build the sample set
    samples_list = get_sample_sets(pop_ids, node_ids, ns_list_copy, paired_pops, sample_rng)

    sfs_ns = None
    sfs_syn = None

    for rep_id in range(1, n_reps + 1):
        path = f"{base_path}_ID_{rep_id}.trees"
        ts = tskit.load(path)
        nts = msprime.sim_mutations(ts, rate=(1/3.5)*Q*7e-9,
                                    model=msprime.SLiMMutationModel(type=3),
                                    random_seed=mut_seeds[rep_id-1], keep=False)

        rep_ns  = ts.allele_frequency_spectrum(sample_sets=samples_list, polarised=True, span_normalise=False)
        rep_syn = nts.allele_frequency_spectrum(sample_sets=samples_list, polarised=True, span_normalise=False)

        sfs_ns  = rep_ns  if sfs_ns  is None else sfs_ns  + rep_ns
        sfs_syn = rep_syn if sfs_syn is None else sfs_syn + rep_syn
        
        del ts, nts  # free memory before loading the next tree sequences
    # convert to a dadi Spectrum object and return
    return dadi.Spectrum(sfs_syn), dadi.Spectrum(sfs_ns)

def get_bootstrap_spectra(ts_path, ns_list, n_reps, num_windows, num_bootstraps, Q=5,
                          mut_seed=123, sampling_seed=42, window_seed=4242, paired_pops=None):
    """
    Calculate boostrap SFS from a tree sequence.

    ts_path: a list of tskit tree sequence objects from SLiM (just non-synonymous mutations)
    ns_list: a list of sample sizes (in number of haplotypes) for each "population" (or subgenome)
        this matches the dadi convention... here, sampling 4 haplotypes is equivalent to sampling 1 autotetraploid
    n_reps: number of replicates to load from the ts_path
    num_windows: number of windows to split one tree sequence into
        *note: this should be the number of windows we want over 1/10 of the genome
        so, to get a reasonable chunk size (~3e5) we want num_windows = 2 for a genome of length 5e5,
    num_bootstraps: number of bootstrap spectra to generate
    Q: scaling factor from the SLiM simulation
    mut_seed: seed for the random number generator for overlaying neutral mutations
    sampling_seed: seed for the random number generator for sampling individuals
    window_seed: seed for the random number generator for sampling windows
    paired_pops: list of lists of population ids whose individuals must be sampled together
        (see get_sfs_from_ts). None (default) samples every population independently.

    Returns:
    sfs_syn: a dadi Spectrum object for only neutral mutations
    """
    rep_id = get_rep_from_filepath(ts_path)
    # set up a reproducible rng for overlaying neutral mutations
    mut_rng = np.random.default_rng(mut_seed + rep_id*100)
    mut_seeds = mut_rng.integers(1, 2**32, size=n_reps)

    # convert the sizes of the sfs to be in number of individuals
    ns_list_copy = [int(ns/2) for ns in ns_list]
    if any(ns % 2 != 0 for ns in ns_list):
        raise ValueError("ns must be a multiple of 2.")

    # get the base path from the ts_path
    base_path = re.sub(r'_ID_\d+\.trees$', '', ts_path)
    # use rep 1 to establish sample sets
    ts0 = tskit.load(f"{base_path}_ID_1.trees")
    # extract the nodes and populations from the ts
    node_ids = ts0.individuals_nodes
    pop_ids = ts0.individuals_population
    # set up a reproducible rng for sampling
    sample_rng = np.random.default_rng(sampling_seed)
    # build the sample set
    samples_list = get_sample_sets(pop_ids, node_ids, ns_list_copy, paired_pops, sample_rng)

    # since we most often want many bootstraps, it is much more efficient to
    # calculate the SFS for each window once and then sample from those SFS
    windows = np.linspace(0, ts0.sequence_length, num_windows+1)
    windowed_spectra_ns = []
    windowed_spectra_syn = []

    # calculate the windowed SFS
    for rep_id in range(1, n_reps + 1):
        path = f"{base_path}_ID_{rep_id}.trees"
        ts = tskit.load(path)
        nts = msprime.sim_mutations(ts, rate=(1/3.5)*Q*7e-9,
                                    model=msprime.SLiMMutationModel(type=3),
                                    random_seed=mut_seeds[rep_id-1], keep=False)

        # calculate the sfs for the non-synonymous mutations
        windowed_sfs_ns = ts.allele_frequency_spectrum(sample_sets=samples_list, polarised=True, span_normalise=False, windows=windows)
        # calculate the sfs for ALL mutations
        windowed_sfs_syn = nts.allele_frequency_spectrum(sample_sets=samples_list, polarised=True, span_normalise=False, windows=windows)
        # append the windowed SFS to the lists one at a time with extend
        windowed_spectra_ns.extend(windowed_sfs_ns)
        windowed_spectra_syn.extend(windowed_sfs_syn)

    # then, we can calculate the bootstrap SFS by sampling from the windowed SFS
    bootstrap_spectra_ns = []
    bootstrap_spectra_syn = []
    rng = np.random.default_rng(window_seed + rep_id*100)
    # we want to sample randomly over all the windows from the whole genome
    # so, we sample num_windows*len(ts_list) windows
    total_windows = num_windows*n_reps

    for i in range(num_bootstraps):
        window_indices = rng.integers(0, total_windows, size=total_windows)
        # create empty sfs
        boot_sfs_ns = np.zeros(windowed_spectra_ns[0].shape)
        boot_sfs_syn = np.zeros(windowed_spectra_syn[0].shape)
        # add each windowed sfs to the empty sfs
        for j in window_indices:
            boot_sfs_ns += windowed_spectra_ns[j]
            boot_sfs_syn += windowed_spectra_syn[j]
        # convert to dadi spectrum objects
        boot_sfs_ns = dadi.Spectrum(boot_sfs_ns)
        boot_sfs_syn = dadi.Spectrum(boot_sfs_syn)
        # append the bootstrap sfs to the lists
        bootstrap_spectra_ns.append(boot_sfs_ns)
        bootstrap_spectra_syn.append(boot_sfs_syn)

    return bootstrap_spectra_ns, bootstrap_spectra_syn


# utility function for wrapping dadi demographic model functions 
# this is not necessary if we are "collapsing" the SFS for an autotetraploid from SLiM
def wrap_model_collapsed(model_func, collapsed_ids, original_ns, ploidyType='allotetraploid'):
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
        For allotetraploids: this is the ns passed to the model
        For autotetraploids: this is the original uncollapsed ns
    ploidyType : str
        'autotetraploid' or 'allotetraploid'
        For autotetraploids, the original_ns will be adjusted to sum collapsed populations
    """
    def wrapped_model_func(params, ns, pts):
        # For allotetraploids, we need to use the original_ns for the model
        if ploidyType == 'allotetraploid':
            # allotetraploid: use original_ns for the model
            fs = model_func(params, original_ns, pts)
            # Then collapse the resulting spectrum
            fs = fs.combine_pops(collapsed_ids)
        else:
            # autotetraploid: use ns for the model
            fs = model_func(params, ns, pts)
        return fs
        
    wrapped_model_func.__param_names__ = model_func.__param_names__
    return wrapped_model_func

# optimization functions
def opt_single_spectrum(input_file, model_func, initial_params, lower_bounds, upper_bounds,
                        ns_list, ploidyType, num_reps, mut_seed=123, sampling_seed=42,
                        folded=False, collapsed_ids=None, misid=False,
                        fixed_params=None, num_opts=100, pts_l=[101, 111, 121],
                        output_dir="inference_results", maxeval=600, ll_tol=0.0001,
                        algorithm=nlopt.LN_BOBYQA, func_tol=1e-6, param_tol=1e-6,
                        paired_pops=None):
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
    ns_list : list
        Sample sizes for each "population" in number of haploids (so must be divisible by 2)
        this matches the dadi convention... here, sampling 4 haplotypes is equivalent to sampling 1 autotetraploid
    paired_pops : list
        List of lists of population ids whose individuals must be sampled together
        (e.g. [[2, 3]] for an autotetraploid split across populations 2 and 3). None
        (default) samples every population independently.
    ploidyType : str
        'autotetraploid' or 'allotetraploid'
    num_reps : int  
        number of replicate tree sequences to iteratively load and calculate the SFS for
    mut_seed : int
        Seed for the random number generator for overlaying neutral mutations
    sampling_seed : int
        Seed for the random number generator for sampling individuals
    folded : bool
        Whether to fold the SFS (default: False)
    collapsed_ids : list
        Population IDs to collapse (None for no collapsing)
    misid : bool
        Whether to include ancestral state misidentification (default: False)
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
    #try:
    Q = get_Q_from_filepath(input_file)
    sfs_syn, sfs_ns = get_sfs_from_ts(input_file, ns_list, num_reps, Q=Q, mut_seed=mut_seed, sampling_seed=sampling_seed, paired_pops=paired_pops)
    # for demographic inference, we only need sfs_syn
    sfs = sfs_syn
    #except Exception as e:
        # return {
        #     'success': False,
        #     'filename': input_file,
        #     'output_file': None,
        #     'error': f"Error loading tree sequence: {str(e)} and initializing SFS."
        # }

    # add misidentification if desired
    if misid:
        param_names = list(model_func.__param_names__) + ["misid"]
        model_func = dadi.Numerics.make_anc_state_misid_func(model_func)
        model_func.__param_names__ = param_names
    # Fold and collapse if desired
    if collapsed_ids is not None:
        model_func = wrap_model_collapsed(model_func, collapsed_ids, ns_list, ploidyType)
        sfs = sfs.combine_pops(collapsed_ids)
        sfs_ns = sfs_ns.combine_pops(collapsed_ids)
    if folded:
        sfs = sfs.fold()
        sfs_ns = sfs_ns.fold()
    

    extrap_func = dadi.Numerics.make_extrap_func(model_func)

    # Create output file
    base_name = os.path.basename(input_file).replace('.trees', '')
    output_file = os.path.join(output_dir, f"optimization_{base_name}.txt")

    sfs.to_file(os.path.join(output_dir, f"{base_name}.sfs"))
    sfs_ns.to_file(os.path.join(output_dir, f"nonsyn{base_name}.sfs"))

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
            if i >= 10: # run at least 10 optimizations before checking convergence
                ll_list.sort()
                print(ll_list[0], ll_list[1], ll_list[2])
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
                                                    verbose=0,
                                                    fixed_params=fixed_params, 
                                                    ftol_abs = func_tol, 
                                                    xtol_abs = param_tol)
            ll_list.append(np.abs(ll_model))

            model_fs = extrap_func(popt, sfs.sample_sizes, pts_l)
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
            'bottlegrowth_dip_size_change': auto_demographics.bottlegrowth_dip_size_change,
            'bottlegrowth_dip_size_change_mig': auto_demographics.bottlegrowth_dip_size_change_mig,
            'bottlegrowth_dip_size_change_asym_mig': auto_demographics.bottlegrowth_dip_size_change_asym_mig,
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
        ns_list = config['ns_list']
        num_reps = config['num_reps']

        # Optional fields with defaults
        folded = config.get('folded', False)
        collapsed_ids = config.get('collapsed_ids', None)
        misid = config.get('misid', False)
        fixed_params = config.get('fixed_params', None)
        num_opts = config.get('num_opts', 100)
        pts_l = config.get('pts_l', [101, 111, 121])
        maxeval = config.get('maxeval', 600)
        algorithm = config.get('algorithm', nlopt.LN_BOBYQA)
        ll_tol = config.get('ll_tol', .0001)
        func_tol = config.get('func_tol', 1e-6)
        param_tol = config.get('param_tol', 1e-6)
        mut_seed = config.get('mut_seed', 123)
        sampling_seed = config.get('sampling_seed', 42)
        paired_pops = config.get('paired_pops', None)

        output_dict = opt_single_spectrum(args.input_file, model_func, initial_params,
                                          lower_bounds, upper_bounds,
                                          ns_list, ploidyType, num_reps,
                                          mut_seed, sampling_seed,
                                          folded, collapsed_ids, misid,
                                          fixed_params, num_opts, pts_l,
                                          output_dir, maxeval, ll_tol, algorithm, func_tol, param_tol,
                                          paired_pops=paired_pops)

        print(f"Finished optimization for {output_dict['filename']}")
        print(f"Success: {output_dict['success']}")
        print(f"Optimization results saved to {output_dict['output_file']}")
        print(f"Error: {output_dict['error']}")
    
if __name__ == '__main__':
    main()