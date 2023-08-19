#!/bin/bash

# Run training on each dataset

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

## Foam
export RING=False
export BATCH_SIZE=16
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3
export NUM_NODES=3
export USE_H5=True


# loop over datasets
NUM_SPARSE_ANGLES_ARRAY=( {10..180..10} )

for NUM_SPARSE_ANGLES in "${NUM_SPARSE_ANGLES_ARRAY[@]}"; do
    echo "Current NUM_SPARSE_ANGLES: $NUM_SPARSE_ANGLES"
    export DATASET_ID=foam_${NUM_SPARSE_ANGLES}ang_1000ex
    echo "Submitting job to train with $DATASET_ID"
    sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5
done
