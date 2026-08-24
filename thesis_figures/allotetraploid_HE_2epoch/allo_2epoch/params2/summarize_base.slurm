#!/bin/bash
#SBATCH --job-name=summarize_base_params2
#SBATCH --account=rgutenk
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1                     
#SBATCH --time=04:00:00
#SBATCH --output=logs/inference_%A.out   
#SBATCH --error=logs/inference_%A.err

mkdir -p logs

# Load required modules and set the venv
module load python/3.11/3.11.4
source /xdisk/rgutenk/sgibbon/dadi_dev_venv3/bin/activate

# Load the tree sequences as an array
BasePath="/xdisk/rgutenk/sgibbon/dadi_msprime_pipeline"
TREES=($BasePath/msprime_tree_sequences/allo_2epoch/params2/*.trees)
# Also load the nested and full inferences
NESTED=($BasePath/dadi_inference_results/allo_2epoch/params2/Full_nested/*.txt)
FULL=($BasePath/dadi_inference_results/allo_2epoch/params2/Full_complex/*.txt)

echo "=========================================="
echo "Start: $(date)"
echo "=========================================="

echo "Running nested uncertainties:"

python3 $BasePath/scripts/summarize_inference_results.py \
    --mode summarize \
    --trees-files "${TREES[@]}" \
    --opt-files "${NESTED[@]}" \
    --output-dir results_base \
    --output-file nested_summary.csv \
    --num-windows 33 \
    --num-bootstraps 100 \
    --seed 42 \
    --nested \
    --config $BasePath/analysis/allo_2epoch/params2/config_base.yaml 

echo "Running full uncertainties:"

python3 $BasePath/scripts/summarize_inference_results.py \
    --mode summarize \
    --trees-files "${TREES[@]}" \
    --opt-files "${FULL[@]}" \
    --output-dir results_base \
    --output-file full_summary.csv \
    --num-windows 33 \
    --num-bootstraps 100 \
    --seed 42 \
    --config $BasePath/analysis/allo_2epoch/params2/config_base.yaml 
    
echo "Running LRT:"

python3 $BasePath/scripts/summarize_inference_results.py \
    --mode lrt \
    --trees-files "${TREES[@]}" \
    --opt-files "${NESTED[@]}" \
    --opt-files-2 "${FULL[@]}" \
    --output-dir results_base \
    --output-file LRT.csv \
    --num-windows 33 \
    --num-bootstraps 100 \
    --seed 42 \
    --config $BasePath/analysis/allo_2epoch/params2/config_base.yaml 

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
