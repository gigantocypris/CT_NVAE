#!/bin/bash
: ${batch_sizes:="8"} 
for batch_size in $batch_sizes; do
    echo $batch_size
    sbatch $CT_NVAE_PATH/slurm/train_single_node.sh batch_size_$batch_size $batch_size $1 $2
done
