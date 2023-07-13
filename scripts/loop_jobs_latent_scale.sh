#!/bin/bash
for i in {4..5}; do
    sbatch $CT_NVAE_PATH/scripts/train_single_node.sh num_latent_scales_$i $i
done
