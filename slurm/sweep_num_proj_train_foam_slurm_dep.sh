#!/bin/bash

# Inputs

export SAVE_NAME=False # Set to False for a new array of jobs, can give array of job IDs to resume
export EPOCHS=500
export NUM_SUBMISSIONS=5 # Max number of submission events
export TIME=24:00:00

export RING_VAL=0
export RING=False
export BATCH_SIZE=16
export SAVE_INTERVAL=1000
export NUM_NODES=3
export USE_H5=True
export DATA_TYPE=foam
export NUM_EXAMPLES=1000
export NUM_SPARSE_ANGLES_ARRAY=( {20..180..20} )
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
echo

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m2859_g
conda activate CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR



# Array to store job IDs
export JOB_ID_ARRAY_ORIG=() # original job ID
export DATASET_ID_ARRAY=()
export JOB_FINAL_ARRAY=()

# loop over datasets
export ind=0
for NUM_SPARSE_ANGLES in "${NUM_SPARSE_ANGLES_ARRAY[@]}"; do
    export JOB_ID_ARRAY=() # store all JOB_IDs for this dataset

    echo "Current NUM_SPARSE_ANGLES: $NUM_SPARSE_ANGLES"
    export PNM=$((10000/$NUM_SPARSE_ANGLES))
    echo "PNM: $PNM"
    export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING_VAL}ring
    export SAVE_NAME_I=${SAVE_NAME[$ind]}
    export ind=$((ind + 1))
    echo "Save name: $SAVE_NAME_I"
    
    
    echo "Submitting job to train with $DATASET_ID"
    export COMMAND="sbatch -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES --time=$TIME $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $SAVE_NAME_I $DATA_TYPE"
    echo "COMMAND: $COMMAND"
    JOB_ID=$(eval "$COMMAND" | awk '{print $4}')
    JOB_ID_ARRAY_ORIG+=("$JOB_ID")
    JOB_ID_ARRAY+=("$JOB_ID")
    export JOB_ID_ORIG=$JOB_ID
    echo "Job ID Original: $JOB_ID_ORIG"

    for ((i = 1; i <= NUM_SUBMISSIONS; i++)); do
        echo "Submitting job to train with $DATASET_ID"

        # This will be run if all the previous jobs do not complete
        export PREVIOUS_JOBS=$(IFS=:; echo "${JOB_ID_ARRAY[*]}")
        export COMMAND_NOTOK="sbatch --dependency=afternotok:$PREVIOUS_JOBS -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES --time=$TIME $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID $EPOCHS $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $JOB_ID_ORIG $DATA_TYPE"

        echo "COMMAND: $COMMAND_NOTOK"
        
        JOB_ID=$(eval "$COMMAND_NOTOK" | awk '{print $4}')
        JOB_ID_ARRAY+=("$JOB_ID")
        echo "Job ID repeat: $JOB_ID"
    done

    # This will be run (final train and test) after the previous jobs complete
    export COMMAND_ANY="sbatch --dependency=afterany:$JOB_ID -A $NERSC_GPU_ALLOCATION -N $NUM_NODES -n $NUM_NODES --time=$TIME $CT_NVAE_PATH/slurm/train_multi_node_preempt.sh $BATCH_SIZE $CT_NVAE_PATH $DATASET_ID 0 $SAVE_INTERVAL $PNM $RING $NUM_NODES $USE_H5 $JOB_ID_ORIG $DATA_TYPE"
    JOB_ID_ANY=$(eval "$COMMAND_ANY" | awk '{print $4}')
    echo "Job ID analysis: $JOB_ID_ANY"
    JOB_FINAL_ARRAY+=("$JOB_ID_ANY")

    echo "JOB_ID_ARRAY"
    echo "${JOB_ID_ARRAY[@]}"
    echo

    DATASET_ID_ARRAY+=("$DATASET_ID")
done

echo "DATASET_ID_ARRAY"
echo "${DATASET_ID_ARRAY[@]}"

echo "JOB_ID_ARRAY_ORIG"
echo "${JOB_ID_ARRAY_ORIG[@]}"

export INPUT_FILE_ANALYSIS="${JOB_ID_ARRAY_ORIG[0]}.txt"
echo "INPUT_FILE_ANALYSIS: $INPUT_FILE_ANALYSIS"
. $CT_NVAE_PATH/slurm/analyze_sweep.sh $JOB_ID_ARRAY_ORIG $INPUT_FILE_ANALYSIS $JOB_FINAL_ARRAY

echo "Script end $(date)";pwd