#!/bin/bash

#SBATCH -J CT_NVAE_ANALYZE       # job name
#SBATCH -L SCRATCH               # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -o %j.out
#SBATCH -e %j.err

export INPUT_FILE=$1
echo "INPUT_FILE=$INPUT_FILE"

module load parallel

echo "jobstart $(date)";pwd
srun parallel --colsep ' ' --jobs 27 $CT_NVAE_PATH/slurm/analyze_sweep_payload.sh {1} {2} {3} {4} {5} :::: $INPUT_FILE
echo "jobend $(date)";pwd