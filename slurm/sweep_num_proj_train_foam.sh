#!/bin/bash

# Inputs

# Change for final evaluation
export SAVE_NAME=False # Set to False for a new array of jobs
# export SAVE_NAME=(14263357 14263361 14263362 14263365 14263367 14263370 14263372 14263375 14263377 14263384 14263388 14263391 14263396 14263400 14263402 14263406 14263410 14263412)
export EPOCHS=1000
export MONITOR_JOBS=True
export TIME=24:00:00
# End of change for final evaluation

export RING=False
export BATCH_SIZE=16
export SAVE_INTERVAL=1000
export PNM=1e3
export NUM_NODES=3
export USE_H5=True
export DATA_TYPE=foam
export NUM_EXAMPLES=1000
export NUM_SPARSE_ANGLES_ARRAY=( {10..180..10} )
export MAX_SUBMISSIONS=10 # Max number of submission events
export SLEEP_TIME=300 # seconds
# See DATASET_ID formatting below

# End of Inputs

# Start script

echo "Script start $(date)";pwd

if [[ $SAVE_NAME = "False" ]]; then
    export SAVE_NAME=()
    # Populate the new array with placeholder values
    for ((i=0; i<${#NUM_SPARSE_ANGLES_ARRAY[@]}; i++)); do
        SAVE_NAME+=(False)
    done
fi

echo "SAVE_NAME: ${SAVE_NAME[@]}"

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
export JOB_ID_ARRAY_ORIG=() # original job ID
export JOB_ID_ARRAY=() # current job ID
export JOB_ID_SUBMITS_LEFT_ARRAY=() # number of submissions for current job ID
export DATASET_ID_ARRAY=()

# loop over datasets
for NUM_SPARSE_ANGLES in "${NUM_SPARSE_ANGLES_ARRAY[@]}"; do
    echo "Current NUM_SPARSE_ANGLES: $NUM_SPARSE_ANGLES"
    export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex
    echo "Submitting job to train with $DATASET_ID"
    JOB_ID=$(sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES --time=$TIME $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $SAVE_NAME $DATA_TYPE | awk '{print $4}')
    JOB_ID_ARRAY+=("$JOB_ID")
    JOB_ID_ARRAY_ORIG+=("$JOB_ID")
    JOB_ID_SUBMITS_LEFT_ARRAY+=("$MAX_SUBMISSIONS")
    DATASET_ID_ARRAY+=("$DATASET_ID")
    echo "Job ID: $JOB_ID"
done

# Print JOB_ID_ARRAY_ORIG 
echo "JOB_ID_ARRAY_ORIG"
echo "${JOB_ID_ARRAY_ORIG[@]}"

# Check a condition
if [[ $MONITOR_JOBS == "False" ]]; then
    echo "Not montioring jobs. Returning."
    return
fi


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
            JOB_ID=$(sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $JOB_ID_ORIGINAL | awk '{print $4}')
            JOB_ID_ARRAY[$i]="$JOB_ID"
            JOB_ID_SUBMITS_LEFT_ARRAY[$i]=$((JOB_ID_SUBMITS_LEFT - 1))
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