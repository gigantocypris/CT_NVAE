#!/bin/bash
for i in {2..3}; do
    sbatch $CT_NVAE_PATH/scripts/train_single_node.sh num_latent_scales_$i $i
done


'''#!/bin/bash

# Define a new script file
SCRIPT_FILE="run_all_jobs.sh"

# Start the new script file with a shebang
echo "#!/bin/bash" > $SCRIPT_FILE

# Loop through your tasks
for i in {2..4}; do
    # Add each task to the new script file
    echo "$CT_NVAE_PATH/scripts/train_single_node.sh num_latent_scales_$i $i" >> $SCRIPT_FILE
done

# Make the new script file executable
chmod +x $SCRIPT_FILE

# Submit the new script file as a single SLURM job
sbatch $SCRIPT_FILE
'''