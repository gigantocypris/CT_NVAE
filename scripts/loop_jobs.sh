FOR i=2 3 4
    sbatch $CT_NVAE_PATH/scripts/train_single_node.sh num_latent_scales_$i $i
