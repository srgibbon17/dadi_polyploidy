#!/bin/bash
#SBATCH --job-name=joint_demog
#SBATCH --array=0-9
#SBATCH --account=rgutenk
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1                     
#SBATCH --time=12:00:00
#SBATCH --output=logs/demog_inference_%A_%a.out   
#SBATCH --error=logs/demog_inference_%A_%a.err

mkdir -p logs

# Load required modules and set the venv
module load python/3.11/3.11.4
source /xdisk/rgutenk/sgibbon/dadi_dev_venv3/bin/activate

# Load the tree sequences as an array
BasePath="/xdisk/rgutenk/sgibbon/SLiM/High_r_production/Autotetraploids/Joint_DFE_migration"

# Simulation path
SimPath="${BasePath}/simulations2/trees/final"

# only load tree sequences which have ID=1
TREES=($SimPath/*ID_1.trees)

echo "=========================================="
echo "Start: $(date)"
echo "=========================================="

echo "Running inference on ${TREES[$SLURM_ARRAY_TASK_ID]}"

python3 $BasePath/demography2/dadi_demographic_inference.py \
    --config-file demog_configs.yaml \
    --input-file ${TREES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
