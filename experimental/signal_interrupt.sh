#!/bin/bash

#SBATCH -A m3562_g
#SBATCH --signal=SIGINT@60
#SBATCH --time=3
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -J sig_interrupt      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=32
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --requeue
#SBATCH --open-mode=append

# this needs 300 * 5s = 5 min to complete

echo "jobstart $(date)";pwd
python $SCRATCH/CT_NVAE/experimental/test_signal_interrupt.py
echo "jobend $(date)";pwd