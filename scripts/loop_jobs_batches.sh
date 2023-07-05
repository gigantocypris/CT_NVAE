#!/bin/bash
: ${batch_sizes:="8"} 
for i in $batch_sizes; do
    sbatch $CT_NVAE_PATH/scripts/train_single_node_batch.sh batch_size_$i $i $1 $CT_NVAE_PATH
done