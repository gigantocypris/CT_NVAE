#!/bin/bash
: ${value_array:="8"} 
for value in $value_array; do
    echo $value
    sbatch -A $NERSC_GPU_ALLOCATION -t 00:30:00 $CT_NVAE_PATH/slurm/train_single_node.sh $value $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL
done
