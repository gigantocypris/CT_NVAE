#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:05:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err

module load python
conda activate tomopy

export EXPR_ID=$1

export BATCH_SIZE=8
export EPOCHS=100
export NUM_LATENT_SCALES=$2
export NUM_GROUPS_PER_SCALE=10
export NUM_POSTPROCESS_CELLS=2
export NUM_PREPROCESS_CELLS=2
export NUM_CELL_PER_COND_ENC=2
export NUM_CELL_PER_COND_DEC=2
export NUM_LATENT_PER_GROUP=3
export NUM_PREPROCESS_BLOCKS=2
export NUM_POSTPROCESS_BLOCKS=2
export WEIGHT_DECAY_NORM=1e-2
export NUM_CHANNELS_ENC=4
export NUM_CHANNELS_DEC=4
export NUM_NF=0
export PNM=1e1


export CHECKPOINT_DIR=checkpts

echo "jobstart $(date)";pwd

python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $EXPR_ID --dataset foam --batch_size $BATCH_SIZE --epochs $EPOCHS --num_latent_scales $NUM_LATENT_SCALES --num_groups_per_scale $NUM_GROUPS_PER_SCALE --num_postprocess_cells $NUM_POSTPROCESS_CELLS --num_preprocess_cells $NUM_PREPROCESS_CELLS --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC --num_latent_per_group $NUM_LATENT_PER_GROUP --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS --weight_decay_norm $WEIGHT_DECAY_NORM --num_channels_enc $NUM_CHANNELS_ENC --num_channels_dec $NUM_CHANNELS_DEC --num_nf 0 --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm $PNM
        
echo "jobend $(date)";pwd