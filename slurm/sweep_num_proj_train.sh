#!/bin/bash

# Inputs
export RING=False
export BATCH_SIZE=16
export EPOCHS=100000
export SAVE_INTERVAL=1000
export PNM=1e3
export NUM_NODES=3
export USE_H5=True
export SAVE_NAME=False


NUM_SPARSE_ANGLES_ARRAY=( {10..180..10} )
export MAX_SUBMISSIONS=10 # Max number of submission events
export SLEEP_TIME=300 # seconds
# See DATASET_ID below

# End of Inputs

# Start script

echo "Script start $(date)";pwd

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



# Array to store job IDs
JOB_ID_ARRAY_ORIG=() # original job ID
JOB_ID_ARRAY=() # current job ID
JOB_ID_SUBMITS_LEFT_ARRAY=() # number of submissions for current job ID
DATASET_ID_ARRAY=()

# loop over datasets
for NUM_SPARSE_ANGLES in "${NUM_SPARSE_ANGLES_ARRAY[@]}"; do
    echo "Current NUM_SPARSE_ANGLES: $NUM_SPARSE_ANGLES"
    export DATASET_ID=foam_${NUM_SPARSE_ANGLES}ang_1000ex
    echo "Submitting job to train with $DATASET_ID"
    JOB_ID=$(sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $SAVE_NAME | awk '{print $4}')
    JOB_ID_ARRAY+=("$JOB_ID")
    JOB_ID_ARRAY_ORIG+=("$JOB_ID")
    JOB_ID_SUBMITS_LEFT_ARRAY+=("$MAX_SUBMISSIONS")
    DATASET_ID_ARRAY+=("$DATASET_ID")
    echo "Job ID: $JOB_ID"
done


# Monitor and resubmit loop
while [[ ${#JOB_ID_ARRAY[@]} -gt 0 ]]; do
    # Wait for some time before checking again (adjust as needed)
    sleep $SLEEP_TIME

    # Loop through job IDs array
    for ((i=0; i<${#JOB_ID_ARRAY[@]}; i++)); do
        JOB_ID_ORIGINAL=${JOB_ID_ARRAY_ORIG[$i]}
        JOB_ID=${JOB_ID_ARRAY[$i]}
        JOB_ID_SUBMITS_LEFT=${JOB_ID_SUBMITS_LEFT_ARRAY[$i]}
        DATASET_ID=${DATASET_ID_ARRAY[$i]}
        echo "Checking job $JOB_ID for dataset $DATASET_ID"

        # Check status of the job
        export JOB_STATUS=$(squeue -u vidyagan -h -t pending,running -j "$JOB_ID")

        # Resubmit if stopped and submission events remaining
        if [[ -z $JOB_STATUS && $JOB_ID_SUBMITS_LEFT -gt 0 ]]; then
            echo "Job $JOB_ID stopped, original job is $JOB_ID_ORIGINAL. Resubmitting with $JOB_ID_SUBMITS_LEFT submits left..."
            JOB_ID_SUBMITS_LEFT=$((MAX_SUBMISSIONS - 1))
            JOB_ID=$(sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $JOB_ID_ORIGINAL | awk '{print $4}')
            JOB_ID_ARRAY[$i]="$JOB_ID"
            JOB_ID_SUBMITS_LEFT_ARRAY[$i]=$JOB_ID_SUBMITS_LEFT
            echo "New job ID: $JOB_ID"
        else
            # Remove job ID from array if it's done and there are no submission events left
            if [[ -z $JOB_STATUS && $JOB_ID_SUBMITS_LEFT -eq 0 ]]; then
                echo "Job $JOB_ID done and no submission events left. Removing from array."
                unset JOB_ID_ARRAY[$i]
                unset JOB_ID_ARRAY_ORIG[$i]
                unset JOB_ID_SUBMITS_LEFT_ARRAY[$i]
                unset DATASET_ID_ARRAY[$i]
            fi
        fi
    done
done

echo "Script end $(date)";pwd