# SLiM_DFE_validation
SLiM and python scripts for validating the DFE inference implemented in dadi.Polyploidy. 

### SLiM_simulations
This folder contains SLiM simulation scripts for generating tree sequences from autotetraploid and allotetraploid models.

### dadi_DFE_inference
This folder contains python scripts for inferring demographic and DFE models from the tree sequences output by the SLiM simulations.

### Full simulation to inference pipeline with bash commands
Here, we outline the complete validation pipeline with bash commands beginning from SLiM simulations through DFE inference. The example below is 

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