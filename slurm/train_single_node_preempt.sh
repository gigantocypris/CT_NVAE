#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -n 1            # Number of tasks
#SBATCH --time=24:00:00
#SBATCH -J CT_NVAE      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --signal=B:USR1@60  # sig_time should match your checkpoint overhead time
#SBATCH --requeue
#SBATCH --open-mode=append

export BATCH_SIZE=$1
export CT_NVAE_PATH=$2
export DATASET_ID=$3
export EPOCHS=$4
export SAVE_INTERVAL=$5
export PNM=$6
export RING=$7

export NUM_LATENT_SCALES=2
export NUM_GROUPS_PER_SCALE=10
export NUM_POSTPROCESS_CELLS=3
export NUM_PREPROCESS_CELLS=3
export NUM_CELL_PER_COND_ENC=2
export NUM_CELL_PER_COND_DEC=2
export NUM_LATENT_PER_GROUP=20
export NUM_PREPROCESS_BLOCKS=2
export NUM_POSTPROCESS_BLOCKS=2
export WEIGHT_DECAY_NORM=1e-2
export NUM_CHANNELS_ENC=32
export NUM_CHANNELS_DEC=32
export NUM_NF=0

echo "jobstart $(date)";pwd

srun --cpus-per-task=32 python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $SLURM_JOB_ID --dataset $DATASET_ID --batch_size $BATCH_SIZE --epochs $EPOCHS --num_latent_scales $NUM_LATENT_SCALES --num_groups_per_scale $NUM_GROUPS_PER_SCALE --num_postprocess_cells $NUM_POSTPROCESS_CELLS --num_preprocess_cells $NUM_PREPROCESS_CELLS --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC --num_latent_per_group $NUM_LATENT_PER_GROUP --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS --weight_decay_norm $WEIGHT_DECAY_NORM --num_channels_enc $NUM_CHANNELS_ENC --num_channels_dec $NUM_CHANNELS_DEC --num_nf 0  --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm $PNM --save_interval $SAVE_INTERVAL --cont_training --model_ring_artifact $RING

echo "jobend $(date)";pwd


