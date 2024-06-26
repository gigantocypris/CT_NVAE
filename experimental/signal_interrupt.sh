#!/bin/bash

#SBATCH -A m3562_g
#SBATCH --time=00:06:00
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J sig_interrupt      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -q debug_preempt
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --signal=USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

echo "jobstart $(date)";pwd
srun --cpus-per-task=128 python $SCRATCH/CT_NVAE/experimental/test_signal_interrupt.py
echo "jobend $(date)";pwd