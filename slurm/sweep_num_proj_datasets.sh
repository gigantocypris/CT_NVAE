#!/bin/bash

# Create the datasets

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m3562_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR


export RANDOM_ANGLES=True
export RING=0
export ALGORITHM=gridrec
export DO_PART_ONE=True
export DO_PART_TWO=False
export DATA_TYPE=covid
export NUM_EXAMPLES=650
export IMAGE_ID=${DATA_TYPE}_${NUM_EXAMPLES}ex
export DO_PART_ONE=False
export DO_PART_TWO=True


# loop over different numbers of angles
NUM_SPARSE_ANGLES_ARRAY=( {10..180..10} )

for NUM_SPARSE_ANGLES in "${NUM_SPARSE_ANGLES_ARRAY[@]}"; do
    echo "Current NUM_SPARSE_ANGLES: $NUM_SPARSE_ANGLES"
    export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_1000ex
    echo "Submitting job to create $DATASET_ID"
    sbatch --time=02:00:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
done





