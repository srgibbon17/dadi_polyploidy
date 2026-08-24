# msprime_demography_validation
Code for simulating genealogies with a nonequilibrium demographic model and inferring the demographic model using the dadi.Polyploidy module.

### Outline of approach to simulation with msprime
Although msprime provides great support for arbitrary ploidy, it does not support simultaneous inference of populations of different ploidies (e.g., an autotetraploid with its diploid progenitor). So, instead, we always set `ploidy=2` in the `msprime.sim_ancestry` function and multiply the population sizes for autotetraploids by 2 to get the correct scaling of Ne/drift. 

### Outline of files in this folder

Three primary files which simulate with msprime and infer demographic parameters and uncertainties with dadi. Each of the following files has example command line usage at the top of the file (also see below):
- `generate_msprime_samples.py`: Simulates tree sequences with mutations under a given demographic model and saves the resulting tree sequence.
- `dadi_run_inference.py`: Runs demographic inference with dadi on the SFS from the tree sequence. 
- `summarize_inference_results.py`: Computes uncertainties, adjusted LRTs, and composite AIC/BIC using bootstrap SFS and dadi.GIM and then summarizes the point estimates and uncertainties into a csv file.

Each of these files has a corresponding YAML configuration file which specifies the demographic model to infer and other parameters for simulation and inference.
- `demography_configs_ex.yaml`: Example YAML configuration file for simulating with msprime.
- `inference_configs_ex.yaml`: Example YAML configuration file for demographic inference with dadi.
- `summary_configs_ex.yaml`: Example YAML configuration file for summarizing inference results and computing uncertainties/LRTs/AIC/etc.

Finally, two additional files which store additional demographic models for dadi:
- `auto_demographic_functions.py`: Additional dadi demographic model functions for autotetraploids
- `allo_demographic_functions.py`: Additional dadi demographic model functions for allotetraploids

In general, the pipeline is: 
1. Simulate mutated tree sequences with msprime (`generate_msprime_samples.py`)
2. Compute the SFS and infer demographic parameters with dadi (`dadi_run_inference.py`)
3. Summarize the inference results and compute uncertainties/LRTs/AIC/etc. (`summarize_inference_results.py`)

### Snakemake workflow
A Snakemake workflow that runs this pipeline end-to-end (steps 1-3 above, for the allo_2epoch nested-vs-full-model/LRT example) lives at [`../workflows/msprime_demography_validation`](../workflows/msprime_demography_validation). Scale (replicates, grid points, number of optimizations, bootstrap settings) is controlled entirely by the YAML files in that workflow's `config/` directory. Run it with:
```bash
cd ../workflows/msprime_demography_validation
conda run -n dadi-dev snakemake -s Snakefile --cores 4 -p
```
This has been tested locally at a small scale (10 replicates) and completes in a few minutes; scale up `replicates` and the other config values for a production run. Note that the workflow's tree-sequence filename templating currently only supports the `allo_2epoch` model; extending it to the other models in `generate_msprime_samples.py` just requires adding their filename templates to `trees_filename()` in the Snakefile.

#### Example command line usage
First, we simulate some tree sequences with msprime (throughout, we assume everything is in a `scripts` folder in the current working directory):
```bash
# Load required modules and set the venv (if on HPC)
module load python/3.11/3.11.4
source /path_to_venv/bin/activate

base_path=$(pwd)

python3 $base_path/scripts/generate_msprime_samples.py \
    --config-file $base_path/scripts/demography_configs_ex.yaml 
```

Then, we infer the demographic model with dadi: 
```bash
# Load required modules and set the venv (if on HPC)
module load python/3.11/3.11.4
source /path_to_venv/bin/activate

base_path=$(pwd)

TREES=($base_path/scripts/params1/*.trees)

python3 $base_path/scripts/dadi_run_inference.py \
    --config-file $base_path/scripts/inference_configs_ex.yaml \
    --input-file ${TREES[$SLURM_ARRAY_TASK_ID]}
```

Finally, we summarize the inference results and compute uncertainties/LRTs/AIC/etc.:
```bash
# Load required modules and set the venv (if on HPC)
module load python/3.11/3.11.4
source /path_to_venv/bin/activate

base_path=$(pwd)

# load tree sequence files and nested and full demographic inference files
TREES=($base_path/scripts/params1/*.trees)
FULL=($base_path/scripts/inference_results/allo_2epoch/params1/Full_model/*.txt)
FULL=($base_path/scripts/inference_results/allo_2epoch/params1/Nested_model/*.txt)

# Run the uncertainties separately for nested and full models
echo "Running nested uncertainties:"

python3 $BasePath/scripts/summarize_inference_results.py \
    --mode summarize \
    --trees-files "${TREES[$SLURM_ARRAY_TASK_ID]}" \
    --opt-files "${NESTED[$SLURM_ARRAY_TASK_ID]}" \
    --output-dir results \
    --output-file nested_summary.csv \
    --num-windows 33 \
    --num-bootstraps 100 \
    --seed 42 \
    --nested \
    --config $base_path/summary_configs_ex.yaml

echo "Running full uncertainties:"

python3 $BasePath/scripts/summarize_inference_results.py \
    --mode summarize \
    --trees-files "${TREES[$SLURM_ARRAY_TASK_ID]}" \
    --opt-files "${FULL[$SLURM_ARRAY_TASK_ID]}" \
    --output-dir results \
    --output-file full_summary.csv \
    --num-windows 33 \
    --num-bootstraps 100 \
    --seed 42 \
    --config $base_path/summary_configs_ex.yaml 
    
# Then, run the LRT
echo "Running LRT:"

python3 $BasePath/scripts/summarize_inference_results.py \
    --mode lrt \
    --trees-files "${TREES[$SLURM_ARRAY_TASK_ID]}" \
    --opt-files "${NESTED[$SLURM_ARRAY_TASK_ID]}" \
    --opt-files-2 "${FULL[$SLURM_ARRAY_TASK_ID]}" \
    --output-dir results \
    --output-file LRT.csv \
    --num-windows 33 \
    --num-bootstraps 100 \
    --seed 42 \
    --config $base_path/summary_configs_ex.yaml 
```