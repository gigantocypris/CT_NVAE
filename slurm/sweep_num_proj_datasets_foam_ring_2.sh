#!/bin/bash

# Create the datasets

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m2859_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR


export RANDOM_ANGLES=True
export RING=0.1
export ALGORITHM=gridrec
export DO_PART_ONE=False
export DO_PART_TWO=True
export DATA_TYPE=foam
export NUM_EXAMPLES=1000
export IMAGE_ID=${DATA_TYPE}_${NUM_EXAMPLES}ex
export DO_PART_ONE=False
export DO_PART_TWO=True


# loop over different numbers of angles
NUM_SPARSE_ANGLES_ARRAY=( {20..180..20} )

for NUM_SPARSE_ANGLES in "${NUM_SPARSE_ANGLES_ARRAY[@]}"; do
    echo "Current NUM_SPARSE_ANGLES: $NUM_SPARSE_ANGLES"
    export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING}ring
    echo "Submitting job to create $DATASET_ID"
    sbatch --time=01:00:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO
done