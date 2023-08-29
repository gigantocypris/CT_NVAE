#!/bin/bash

#SBATCH -J CT_NVAE       # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
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
export NUM_NODES=$8
export USE_H5=$9
export SAVE_NAME=${10}
export DATA_TYPE=${11}

if [ $SAVE_NAME = "False" ]; then
    echo "Setting job name to $SLURM_JOB_ID"
    export SAVE_NAME=$SLURM_JOB_ID
else
    echo "Job name is $SAVE_NAME"
fi

export DATASET_DIR=$WORKING_DIR
export CHECKPOINT_DIR=$WORKING_DIR/checkpts
export MASTER_ADDR=$(hostname)
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

echo $MASTER_ADDR

if [ $DATA_TYPE = "foam" ]; then
    echo "Using foam data parameters"
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
    export MIN_GROUPS_PER_SCALE=1

    echo "jobstart $(date)";pwd

    srun --cpus-per-task=128 python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $SAVE_NAME --dataset $DATASET_ID --batch_size $BATCH_SIZE --epochs $EPOCHS --num_latent_scales $NUM_LATENT_SCALES --num_groups_per_scale $NUM_GROUPS_PER_SCALE --num_postprocess_cells $NUM_POSTPROCESS_CELLS --num_preprocess_cells $NUM_PREPROCESS_CELLS --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC --num_latent_per_group $NUM_LATENT_PER_GROUP --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS --weight_decay_norm $WEIGHT_DECAY_NORM --num_channels_enc $NUM_CHANNELS_ENC --num_channels_dec $NUM_CHANNELS_DEC --num_nf $NUM_NF  --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm $PNM --save_interval $SAVE_INTERVAL --cont_training --model_ring_artifact $RING --num_proc_node $NUM_NODES --use_h5 $USE_H5 --min_groups_per_scale $MIN_GROUPS_PER_SCALE --final_test

    echo "jobend $(date)";pwd
else
    echo "Using brain or covid data parameters"
    export NUM_LATENT_SCALES=5
    export NUM_GROUPS_PER_SCALE=16
    export NUM_POSTPROCESS_CELLS=2
    export NUM_PREPROCESS_CELLS=2
    export NUM_CELL_PER_COND_ENC=2
    export NUM_CELL_PER_COND_DEC=2
    export NUM_LATENT_PER_GROUP=20
    export NUM_PREPROCESS_BLOCKS=1
    export NUM_POSTPROCESS_BLOCKS=1
    export WEIGHT_DECAY_NORM=1e-2
    export NUM_CHANNELS_ENC=30
    export NUM_CHANNELS_DEC=30
    export NUM_NF=2
    export MIN_GROUPS_PER_SCALE=4
    export WEIGHT_DECAY_NORM_INIT=1.

    echo "jobstart $(date)";pwd

    srun --cpus-per-task=128 python $CT_NVAE_PATH/train.py --root $CHECKPOINT_DIR --save $SAVE_NAME --dataset $DATASET_ID --batch_size $BATCH_SIZE --epochs $EPOCHS --num_latent_scales $NUM_LATENT_SCALES --num_groups_per_scale $NUM_GROUPS_PER_SCALE --num_postprocess_cells $NUM_POSTPROCESS_CELLS --num_preprocess_cells $NUM_PREPROCESS_CELLS --num_cell_per_cond_enc $NUM_CELL_PER_COND_ENC --num_cell_per_cond_dec $NUM_CELL_PER_COND_DEC --num_latent_per_group $NUM_LATENT_PER_GROUP --num_preprocess_blocks $NUM_PREPROCESS_BLOCKS --num_postprocess_blocks $NUM_POSTPROCESS_BLOCKS --weight_decay_norm $WEIGHT_DECAY_NORM --num_channels_enc $NUM_CHANNELS_ENC --num_channels_dec $NUM_CHANNELS_DEC --num_nf $NUM_NF  --ada_groups --num_process_per_node 4 --use_se --res_dist --fast_adamax --pnm $PNM --save_interval $SAVE_INTERVAL --cont_training --model_ring_artifact $RING --num_proc_node $NUM_NODES --use_h5 $USE_H5 --min_groups_per_scale $MIN_GROUPS_PER_SCALE --weight_decay_norm_anneal --weight_decay_norm_init $WEIGHT_DECAY_NORM_INIT --final_test

    echo "jobend $(date)";pwd
fi




