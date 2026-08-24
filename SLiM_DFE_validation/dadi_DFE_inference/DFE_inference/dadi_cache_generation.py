"""
Generate dadi DFE caches for polyploid inference with selection.
Takes a nonsyn SFS file and optimal demographic parameters to generate 1D and/or 2D caches.

Example command line usage: 

python3 $path_to_script/dadi_cache_generation.py \
    --nonsyn-file path_to_nonsyn_SFS_file \
    --opt-file path_to_demographic_inference_optimization_results_file \
    --ploidyType autotetraploid \
    --model bottleneck_asym_mig_w_dips \
    --cache-type both \
    --output-dir path_to_output_directory \
    --gamma-bounds 1e-4 2000 \
    --gamma-pts 50 \
    --cpus 20

Explanation of command line arguments:
    nonsyn-file: Path to nonsynonymous SFS file (*.sfs) (required, used to get sample sizes)
    opt-file: Path to optimization results file from demographic inference (required)
    ploidyType: Type of polyploid ('autotetraploid', 'allotetraploid') (required)
    model: Demographic model name (required); see list of available models below
    cache-type: Type(s) of cache to generate (default: 1d) (optional, can be '1d', '2d', or 'both')
    output-dir: Output directory for cache files (defaults to same dir as nonsyn_file) (optional)
    gamma-bounds: Lower and upper bounds for gamma=2Ns (population-scaled selection coefficient) (default: 1e-2 20) (optional)
    gamma-pts: Number of gamma points to sample (default: 5 for speed, but more should be used) (optional)
    cpus: Number of CPUs to use (default: 1) (optional)
"""

import dadi
import dadi.DFE as DFE
import os
import argparse
import pickle
import time

def get_dfe_model_function(ploidyType, model, cache_type='1d'):
    """
    Get DFE model function with selection.

    Parameters:
    -----------
    ploidyType : str
        'autotetraploid', 'allotetraploid', etc.
    model : str
        Model name (e.g., 'two_epoch', 'bottlegrowth')
    cache_type : str
        '1d' for single gamma or non-joint DFE or '2d' for independent gammas in a joint DFE

    Returns:
    --------
    model_func : function
        DFE demographic model function
    """

    if ploidyType == 'autotetraploid':
        import dadi.Polyploidy.auto_demographics_sel as auto_dfe

        model_map_1d = {
            # 1D DFE (non-joint)
            'two_epoch': auto_dfe.two_epoch_sel,
            'bottlegrowth': auto_dfe.bottlegrowth_sel,
            'three_epoch': auto_dfe.three_epoch_sel,
            # joint DFE with single gamma
            'bottleneck_w_dips': auto_dfe.bottleneck_w_dips_sel_single_gamma,
            'bottleneck_mig_w_dips': auto_dfe.bottleneck_mig_w_dips_sel_single_gamma,
            'bottleneck_asym_mig_w_dips': auto_dfe.bottleneck_asym_mig_w_dips_sel_single_gamma,
            'bottlegrowth_w_dips': auto_dfe.bottlegrowth_w_dips_sel_single_gamma,
            'bottlegrowth_mig_w_dips': auto_dfe.bottlegrowth_mig_w_dips_sel_single_gamma,
            'bottlegrowth_asym_mig_w_dips': auto_dfe.bottlegrowth_asym_mig_w_dips_sel_single_gamma,
            'bottlegrowth_dip_size_change': auto_dfe.bottlegrowth_dip_size_change_sel_single_gamma,
            'bottlegrowth_dip_size_change_mig': auto_dfe.bottlegrowth_dip_size_change_mig_sel_single_gamma,
            'bottlegrowth_dip_size_change_asym_mig': auto_dfe.bottlegrowth_dip_size_change_asym_mig_sel_single_gamma
        }

        model_map_2d = {
            # joint DFE with independent gammas
            'bottleneck_w_dips': auto_dfe.bottleneck_w_dips_sel,
            'bottleneck_mig_w_dips': auto_dfe.bottleneck_mig_w_dips_sel,
            'bottleneck_asym_mig_w_dips': auto_dfe.bottleneck_asym_mig_w_dips_sel,
            'bottlegrowth_w_dips': auto_dfe.bottlegrowth_w_dips_sel,
            'bottlegrowth_mig_w_dips': auto_dfe.bottlegrowth_mig_w_dips_sel,
            'bottlegrowth_asym_mig_w_dips': auto_dfe.bottlegrowth_asym_mig_w_dips_sel,
            'bottlegrowth_dip_size_change': auto_dfe.bottlegrowth_dip_size_change_sel,
            'bottlegrowth_dip_size_change_mig': auto_dfe.bottlegrowth_dip_size_change_mig_sel,
            'bottlegrowth_dip_size_change_asym_mig': auto_dfe.bottlegrowth_dip_size_change_asym_mig_sel
        }

        if cache_type == '1d':
            return model_map_1d[model]
        else:
            return model_map_2d[model]

    elif ploidyType == 'allotetraploid':
        import allo_demographic_functions_sel as allo_dfe

        model_map_1d = {
            'two_epoch': allo_dfe.two_epoch_sel,
            'bottlegrowth': allo_dfe.bottlegrowth_sel,
            'three_epoch': allo_dfe.three_epoch_sel
        }

        # the allotetraploid models I have implemented do not have a joint DFE
        # because there are no diploids
        model_map_2d = model_map_1d 

        if cache_type == '1d':
            return model_map_1d[model]
        else:
            return model_map_2d[model]
    else:
        raise ValueError(f"Unknown ploidyType: {ploidyType}")


def load_best_params_from_opt_file(opt_file, model_func):
    """
    Load the best parameters from the optimization results file.

    Parameters:
    -----------
    opt_file : str
        Path to optimization results file (optimization_*.txt)
    model_func : function
        Model function with __param_names__ attribute

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
            if ll > best_ll:
                best_ll = ll
                best_line = parts

        # Extract parameters (skip log_likelihood in first column)
        # subtract off the trailing selection coefficient(s) (gamma for 1D,
        # gamma1/gamma2 for 2D), since Cache1D/Cache2D append those
        # themselves and popt should only contain demographic parameters
        num_gammas = sum(1 for p in model_func.__param_names__ if p.startswith('gamma'))
        num_params = len(model_func.__param_names__) - num_gammas
        popt = [float(best_line[i+1]) for i in range(num_params)]

    return popt


def load_sfs_from_file(sfs_file):
    """
    Load a dadi SFS file and extract sample sizes.

    Parameters:
    -----------
    sfs_file : str
        Path to .sfs file

    Returns:
    --------
    fs : dadi.Spectrum
        The spectrum
    ns : list
        Sample sizes
    """
    fs = dadi.Spectrum.from_file(sfs_file)
    ns = list(fs.sample_sizes)
    return fs, ns

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
        else: # autotetraploid
            # otherwise, we can use the ns input as usual
            fs = model_func(params, ns, pts)
        return fs
    wrapped_model_func.__param_names__ = model_func.__param_names__
    return wrapped_model_func

def wrap_model_folded(model_func):
    """
    Wrap a model function to fold the SFS.
    """
    def wrapped_model_func(params, ns, pts):
        fs = model_func(params, ns, pts)
        fs = fs.fold()
        return fs
    wrapped_model_func.__param_names__ = model_func.__param_names__
    return wrapped_model_func

def calculate_pts_l(ns, offset=141):
    """
    Calculate grid points for dadi based on sample sizes.
    For DFE inference with selection, larger grid points are recommended.

    Parameters:
    -----------
    ns : list
        Sample sizes
    offset : int
        Offset to add to max(ns) for grid points
        Default is 141 as recommended for DFE inference

    Returns:
    --------
    pts_l : list
        Three grid point values
    """
    max_ns = max(ns)
    pts_l = [max_ns + offset, max_ns + offset + 10, max_ns + offset + 20]
    return pts_l

def generate_cache(nonsyn_file, opt_file, ploidyType, model,
                   cache_type='1d', output_dir=None,
                   gamma_bounds=[1e-2, 20], gamma_pts=5,
                   pts_l=None, cpus=1, folded=False, collapsed_ids=None):
    """
    Generate a DFE cache for polyploid DFE inference.

    Parameters:
    -----------
    nonsyn_file : str
        Path to nonsynonymous SFS file (*.sfs)
    opt_file : str
        Path to optimization results file from demographic inference
    ploidyType : str
        Type of polyploid ('autotetraploid', 'allotetraploid')
    model : str
        Demographic model name
    cache_type : str
        '1d' for single selection coefficient or '2d' for independent selection coefficients
    output_dir : str
        Output directory for cache files (defaults to same dir as nonsyn_file)
    gamma_bounds : list
        [lower, upper] bounds for gamma (selection coefficient)
    gamma_pts : int
        Number of gamma points to sample
    pts_l : list
        Grid points for dadi (auto-calculated if None)
    cpus : int
        Number of CPUs to use
    folded : bool
        Whether to fold the SFS (default: False)
    collapsed_ids : list
        Population IDs to collapse (None for no collapsing)

    Returns:
    --------
    dict : Status information
    """
    start = time.time()

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(nonsyn_file)
    os.makedirs(output_dir, exist_ok=True)

    
    # Load SFS and extract sample sizes
    fs, ns = load_sfs_from_file(nonsyn_file)

    # Get DFE model function
    demo_sel_model = get_dfe_model_function(ploidyType, model, cache_type)

    # Fold and collapse if desired
    if collapsed_ids is not None:
        demo_sel_model = wrap_model_collapsed(demo_sel_model, collapsed_ids, ns, ploidyType)
        fs = fs.combine_pops(collapsed_ids)
    if folded:
        demo_sel_model = wrap_model_folded(demo_sel_model)

    # Calculate pts_l if not provided
    if pts_l is None:
        pts_l = calculate_pts_l(ns)

    # Load best parameters from optimization file
    popt = load_best_params_from_opt_file(opt_file, demo_sel_model)

    print(f"Loaded parameters: {popt}")
    print(f"Sample sizes: {ns}")
    print(f"Grid points: {pts_l}")
    print(f"Gamma bounds: {gamma_bounds}")
    print(f"Gamma points: {gamma_pts}")

    # Generate cache
    if cache_type == '1d':
        print(f"\nGenerating 1D cache...")
        cache = DFE.Cache1D(popt, ns, demo_sel_model, pts=pts_l,
                            gamma_bounds=gamma_bounds, gamma_pts=gamma_pts,
                            cpus=cpus, verbose=False)
    elif cache_type == '2d':
        print(f"\nGenerating 2D cache...")
        cache = DFE.Cache2D(popt, ns, demo_sel_model, pts=pts_l,
                               gamma_bounds=gamma_bounds, gamma_pts=gamma_pts,
                               cpus=cpus, verbose=False)
    else:
        raise ValueError(f"Unknown cache_type: {cache_type}")

    # Check for negative values
    if (cache.spectra < 0).sum() > 0:
        min_val = cache.spectra.min()
        print(f"\n!!!WARNING!!!")
        print(f"Potentially large negative values!")
        print(f"Most negative value is: {min_val}")
        if min_val < -0.001:
            print(f"Consider rerunning with larger pts_l values")

    # Generate output filename based on input
    base_name = os.path.splitext(os.path.basename(nonsyn_file))[0]
    # Remove 'nonsyn' prefix if present
    if base_name.startswith('nonsyn'):
        base_name = base_name[6:]

    cache_file = os.path.join(output_dir, f"{base_name}_{cache_type}_cache.bpkl")
    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f, protocol=2)

    elapsed = time.time() - start

    return {
        'success': True,
        'cache_file': cache_file,
        'cache_type': cache_type,
        'ns': ns,
        'num_negative': (cache.spectra < 0).sum(),
        'min_value': cache.spectra.min(),
        'elapsed_time': elapsed,
        'error': None
    }

def main():
    parser = argparse.ArgumentParser(
        description='Generate dadi DFE cache for polyploid inference with selection'
    )
    parser.add_argument(
        '--nonsyn-file',
        type=str,
        required=True,
        help='Path to nonsynonymous SFS file (from dadi_demographic_inference.py)'
    )
    parser.add_argument(
        '--opt-file',
        type=str,
        required=True,
        help='Path to optimization results file (optimization_*.txt)'
    )
    parser.add_argument(
        '--ploidyType',
        type=str,
        required=True,
        choices=['autotetraploid', 'allotetraploid'],
        help='Type of polyploid'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Demographic model name'
    )
    parser.add_argument(
        '--cache-type',
        type=str,
        default='1d',
        choices=['1d', '2d', 'both'],
        help='Type(s) of cache to generate (default: 1d)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (defaults to same directory as nonsyn_file)'
    )
    parser.add_argument(
        '--gamma-bounds',
        type=float,
        nargs=2,
        default=[1e-2, 20],
        metavar=('LOWER', 'UPPER'),
        help='Gamma bounds for selection coefficient distribution (default: 1e-2 20)'
    )
    parser.add_argument(
        '--gamma-pts',
        type=int,
        default=5,
        help='Number of gamma points to sample (default: 5, use 50 for publication)'
    )
    parser.add_argument(
        '--pts-l',
        type=int,
        nargs=3,
        default=None,
        metavar=('PTS1', 'PTS2', 'PTS3'),
        help='Grid points for dadi (auto-calculated if not provided)'
    )
    parser.add_argument(
        '--cpus',
        type=int,
        default=1,
        help='Number of CPUs to use (default: 1)'
    )
    parser.add_argument(
        '--folded',
        action='store_true',
        help='Fold the SFS (collapse ancestral/derived)'
    )
    parser.add_argument(
        '--collapsed-ids',
        type=int,
        nargs='+',
        default=None,
        help='Population IDs to collapse (1-indexed, space-separated)'
    )

    args = parser.parse_args()

    # Determine which cache types to generate
    if args.cache_type == 'both':
        cache_types = ['1d', '2d']
    else:
        cache_types = [args.cache_type]

    # Convert pts_l to list if provided
    pts_l = args.pts_l if args.pts_l is not None else None

    # Generate caches
    for cache_type in cache_types:
        print(f"\n{'='*60}")
        print(f"Generating {cache_type.upper()} cache")
        print(f"{'='*60}")

        result = generate_cache(
            nonsyn_file=args.nonsyn_file,
            opt_file=args.opt_file,
            ploidyType=args.ploidyType,
            model=args.model,
            cache_type=cache_type,
            output_dir=args.output_dir,
            gamma_bounds=args.gamma_bounds,
            gamma_pts=args.gamma_pts,
            pts_l=pts_l,
            cpus=args.cpus,
            folded=args.folded,
            collapsed_ids=args.collapsed_ids
        )

        if result['success']:
            print(f"\n✓ Cache generation successful!")
            print(f"  Cache file: {result['cache_file']}")
            print(f"  Sample sizes: {result['ns']}")
            print(f"  Negative values: {result['num_negative']} (min: {result['min_value']:.6e})")
            print(f"  Time elapsed: {result['elapsed_time']:.2f} seconds")
        else:
            print(f"\n✗ Cache generation failed!")
            print(f"  Error: {result['error']}")


if __name__ == '__main__':
    main()
