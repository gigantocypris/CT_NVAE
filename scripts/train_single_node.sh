#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:05:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err

export EXPR_ID=test_0000_slurm
export CHECKPOINT_DIR=checkpts

echo "jobstart $(date)";pwd

python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam \
--batch_size 8 --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 \
--num_postprocess_cells 2 --num_preprocess_cells 2 --num_cell_per_cond_enc 2 \
--num_cell_per_cond_dec 2 --num_latent_per_group 3 --num_preprocess_blocks 2 \
--num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 4 --num_channels_dec 4 \
--num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm 1e1
        
echo "jobend $(date)";pwd