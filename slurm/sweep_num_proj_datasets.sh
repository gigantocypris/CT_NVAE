#!/bin/bash

# Create the datasets

export NUM_EXAMPLES=$1
export RANDOM_ANGLES=$2
export CONSTANT_ANGLES=$3
export TAG=$4
export PNM_NUM=$5
export DATA_TYPE=$6

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m2859_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
mkdir -p $WORKING_DIR
cd $WORKING_DIR

export RING=0
export ALGORITHM=tv
export DO_PART_ONE=False
export DO_PART_TWO=True
export IMAGE_ID=${DATA_TYPE}_${NUM_EXAMPLES}ex
export DO_PART_ONE=False
export DO_PART_TWO=True


# loop over different numbers of angles
NUM_SPARSE_ANGLES_ARRAY=( {20..180..20} )

for NUM_SPARSE_ANGLES in "${NUM_SPARSE_ANGLES_ARRAY[@]}"; do
    echo "Current NUM_SPARSE_ANGLES: $NUM_SPARSE_ANGLES"
    export DATASET_ID=${DATA_TYPE}_${NUM_SPARSE_ANGLES}ang_${NUM_EXAMPLES}ex_${RING}ring_${ALGORITHM}_${RANDOM_ANGLES}random_${CONSTANT_ANGLES}constant${TAG}
    echo "Submitting job to create $DATASET_ID"
    sbatch --time=01:00:00 -A $NERSC_GPU_ALLOCATION $CT_NVAE_PATH/slurm/create_dataset.sh $CT_NVAE_PATH $NUM_EXAMPLES $DATA_TYPE $IMAGE_ID $DATASET_ID $NUM_SPARSE_ANGLES $RANDOM_ANGLES $CONSTANT_ANGLES $RING $ALGORITHM $DO_PART_ONE $DO_PART_TWO $PNM_NUM
done

