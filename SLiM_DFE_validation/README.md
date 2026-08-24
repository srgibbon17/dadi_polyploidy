# SLiM_DFE_validation
SLiM and python scripts for validating the DFE inference implemented in dadi.Polyploidy. 

### SLiM_simulations
This folder contains SLiM simulation scripts for generating tree sequences from autotetraploid and allotetraploid models.

### dadi_DFE_inference
This folder contains python scripts for inferring demographic and DFE models from the tree sequences output by the SLiM simulations.

### Full simulation to inference pipeline with bash commands
Here, we outline the complete validation pipeline with bash commands beginning from SLiM simulations through DFE inference. 

In general, the pipeline is as follows: 
1. Run WF burnin SLiM simulations (*_burn.slim)
2. Run nonWF SLiM simulations with tetraploids (*_final.slim)
3. Run dadi demographic inference (dadi_demographic_inference.py)
4. Run dadi cache generation (dadi_cache_generation.py)
5. Run dadi DFE inference (dadi_DFE_inference.py)

### Snakemake workflow
A Snakemake workflow that runs this pipeline end-to-end (steps 1-5 above, for the autotetraploid-with-diploid-progenitor joint-DFE model: `auto_joint_DFE_burn/finish.slim` + `bottlegrowth_dip_size_change_asym_mig` + lognormal joint mixture DFE) lives at [`../workflows/SLiM_DFE_validation`](../workflows/SLiM_DFE_validation). Scale (Q, L, number of SLiM replicates, REP, cache gamma grid, etc.) is controlled by the YAML files in that workflow's `config/` directory. Run it with:
```bash
cd ../workflows/SLiM_DFE_validation
conda run -n dadi-dev snakemake -s Snakefile --configfile config/config.yaml --cores 4 -p
```
This has been tested locally at a small scale (Q=100, L=1e5, 10 SLiM replicates) and takes roughly an hour, dominated by 2D DFE cache generation; scale up the config values for a production run.

While building and testing this workflow, two pre-existing bugs were found and fixed (unrelated to Snakemake itself, so they also affect the plain bash-command pipeline described below):
- `SLiM_simulations/SLiM_5/auto_joint_DFE_finish.slim` still called the SLiM 4 API (`Individual.genomes`, `Genome.mutationCountsInGenomes`) in several places left over from the SLiM 4->5 port; these are now `.haplosomes` / `mutationCountsInHaplosomes`, matching the fully-converted sibling scripts.
- `dadi_DFE_inference/demographic_inference/dadi_demographic_inference.py`'s `get_sfs_from_ts` and `get_bootstrap_spectra` built the sample node-id set once from the `_ID_1` tree sequence and reused those raw node ids across all `num_reps` replicate tree sequences. Since each replicate is an independent SLiM run with its own node numbering, this silently used invalid/wrong node ids whenever `num_reps > 1` (the documented multi-replicate use case) — it now rebuilds the sample set per replicate, reseeding the sampling RNG identically each time so the sampled ranks stay reproducible.
- Separately, in the `dadi` fork (`dadi/Polyploidy/auto_demographics_sel.py`), `bottlegrowth_dip_size_change_asym_mig_sel_single_gamma` was missing its `__param_names__` attribute because a copy-pasted assignment line accidentally targeted `bottlegrowth_asym_mig_w_dips_sel_single_gamma` instead (and corrupted that function's own attribute in the process); this blocked 1D cache generation for that model and has also been fixed.

A second, separate Snakemake workflow at [`../workflows/SLiM_DFE_validation_auto_bottlegrowth`](../workflows/SLiM_DFE_validation_auto_bottlegrowth) runs the plain single-population autotetraploid case instead (`auto_WF_burn.slim` + `auto_nonWF_finish.slim`, no diploid progenitor tracked past the WGD; `bottlegrowth` demographic model; a plain 1D gamma DFE, no joint/mixture model, so only a 1D cache is needed). It's tested at the same small scale (Q=100, L=1e5, 10 SLiM replicates) and runs in well under 10 minutes, since skipping the 2D mixture cache removes the dominant cost from the other workflow. Run it the same way, from its own directory.

No pipeline bugs were found while testing this second workflow, but one config-tuning pitfall is worth flagging: `dadi_cache_generation.py`'s default `--gamma-pts 5` ("for speed") is too coarse to resolve a gamma DFE's likelihood surface by interpolation — with only 5 grid points spanning a 2000-fold gamma range, the DFE optimizer here ran away toward implausibly large `scale` values pegged at whatever upper bound was set, instead of converging to an interior optimum. Bumping to `--gamma-pts 30` (still only ~10s for a single-population 1D cache) fixed this immediately: log-likelihoods became a well-behaved, tightly-converging unimodal surface, and DFE inference recovered shape=0.266 and scale=180 against the SLiM script's simulated shape=0.27/scale=130 (i.e., a very close shape recovery and a reasonable scale recovery, given only 10 replicates at Q=100 scaling). Use at least `gamma_pts` in the 20-30 range even for quick sanity checks, not the documented speed-oriented default of 5.

#### Run SLiM simulations 
Below, we assume that the relevant SLiM scripts are in the current directory. If not, edit the paths in the commands below.
```bash

# Load SLiM module if on cluster; otherwise, ensure slim is in your path
module load slim/4.3.0

# Run the simulations, preferably in parallel on a cluster
# If simulating multiple replicates with an array job, replace -d REP=1 with -d REP=$SLURM_ARRAY_TASK_ID
for i in {1..10}
do
    (
    slim -d Qfactor=10 -d W=0.925 -d L=5e5 -d BURNIN=20 -d REP=1 -d ID=$i -d SEED=12321 auto_joint_DFE_burn.slim
    echo "WF burn in complete for tree sequence $i at $(date)"
    slim -d Qfactor=10 -d W=0.925 -d RHO=0 -d L=5e5 -d BURNIN=20 -d REP=1 -d ID=$i -d SEED=12321 -d M_RATE=2.0 -d NU1=0.3 -d NU2=0.7 -d TET_TIME=0.3 -d NU_DIP=0.8 -d DIP_TIME=0.15 auto_joint_DFE_finish.slim
    echo "nonWF finish complete for tree sequence $i at $(date)"
    ) &
done

# By defualt, the tree sequences will be saved at trees/burnin and trees/final in the current directory
```

#### Run dadi demographic inference
The below assumes that a `demographic_inference` directory contains the demographic inference script (`dadi_demographic_inference.py`) and the YAML configuration file (`auto_demog_config.yaml`).

```bash

# Load python module and set venv, if applicable
# Note on python venv: should include dadi, tskit, msprime, and the associated dependencies
module load python/3.11/3.11.4 
source /path_to_venv/bin/activate

# set the base path to the working directory
base_path=$(pwd)

# set the path to the final SLiM tree sequences
tree_seq_path=$base_path/trees/final

# load tree sequence files with ID=1 (works with one or multiple replicates)
TREES=($tree_seq_path/*ID_1.trees)

# run the inference
python3 $base_path/demographic_inference/dadi_demographic_inference.py \
    --config-file $base_path/demographic_inference/demog_config.yaml \
    --input-file ${TREES[$SLURM_ARRAY_TASK_ID]} # replace if not on SLURM with relevant array indexing
```

#### Build dadi cache
Starting from the same base path as above, we assume there is a `DFE_inference` directory containing the cache generation script (`dadi_cache_generation.py`), the DFE inference script (`dadi_DFE_inference.py`), and a YAML configuration file (`lognromal_joint_DFE_config.yaml`).

```bash

# Load python module and set venv, if applicable
module load python/3.11/3.11.4
source /path_to_venv/dadi_dev_venv3/bin/activate

# set the base path to the working directory
base_path=$(pwd)

# set the path to the demographic inference results folder
# this includes the output directory specified in the YAML demog_config file
demog_path=$base_path/demographic_inference/bottlegrowth

# load nonsynonymous spectra and optimization results files
SPECTRA=($demog_path/nonsyn*.sfs)
OPT_FILES=($demog_path/optimization*.txt)

# run the cache generation
python3 $base_path/DFE_inference/dadi_cache_generation.py \
    --nonsyn-file ${SPECTRA[$SLURM_ARRAY_TASK_ID]} \
    --opt-file ${OPT_FILES[$SLURM_ARRAY_TASK_ID]} \
    --ploidyType autotetraploid \
    --model bottleneck_asym_mig_w_dips \
    --cache-type both \
    --output-dir caches/bottlegrowth \
    --gamma-bounds 1e-4 2000 \
    --gamma-pts 50 \
    --cpus 20
```

#### Run DFE inference
As for the cache generation, we assume there is a `DFE_inference` directory containing the cache generation script (`dadi_cache_generation.py`), the DFE inference script (`dadi_DFE_inference.py`), and a YAML configuration file (`lognromal_joint_DFE_config.yaml`).

```bash

# Load python module and set venv, if applicable
module load python/3.11/3.11.4
source /path_to_venv/dadi_dev_venv3/bin/activate

# set the base path to the working directory
base_path=$(pwd)

# set the path to the demographic inference results folder
# this includes the output directory specified in the YAML demog_config file
demog_path=$base_path/demographic_inference/bottlegrowth

# set the path to the caches generated above
cache_path=$base_path/DFE_inference/caches/bottlegrowth

# load nonsynonymous spectra and optimization results files
SPECTRA=($demog_path/nonsyn*.sfs)
OPT_FILES=($demog_path/optimization*.txt)

# load the caches
CACHE1D=($cache_path/*1d_cache.bpkl.bpkl)
CACHE2D=($cache_path/*2d_cache.bpkl.bpkl)

# run the DFE inference
python3 $base_path/DFE_inference/dadi_DFE_inference.py \
    --config-file $base_path/DFE_inference/lognormal_joint_DFE_config.yaml \
    --nonsyn-sfs ${SPECTRA[$SLURM_ARRAY_TASK_ID]} \
    --opt-file ${OPT_FILES[$SLURM_ARRAY_TASK_ID]} \
    --cache1D ${CACHES1D[$SLURM_ARRAY_TASK_ID]} \
    --cache2D ${CACHES2D[$SLURM_ARRAY_TASK_ID]}
```

#### Note on uncertainty analysis with GIM
I have not yet written a script to perform uncertainty analysis with GIM, but this should be relativelt straightforward. The code for sampling bootstrap spectra from the tree sequence is included in the `dadi_demographic_inference.py` script, so the only thing that would need to be extended is passing the relevant demographic or DFE model function to `dadi.Godambe.GIM_uncert` with the bootstrapped SFS from msprime. That would be the next and final step to round out the validation pipeline! 