#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J create_dataset     # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -n 4
#SBATCH -o %j.out
#SBATCH -e %j.err

export CT_NVAE_PATH=$1
export NUM_EXAMPLES=$2
export DATA_TYPE=$3
export IMAGE_ID=$4
export DATASET_ID=$5
export NUM_SPARSE_ANGLES=$6
export RANDOM=$7

export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH

echo "jobstart $(date)";pwd
echo $SLURM_NTASKS

if [ $DATA_TYPE = "brain" ]; then
    export CSV_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/brain_merged_info.csv
    export DCM_PATH=/global/cfs/cdirs/m3562/users/hkim/brain_data/raw/stage_2_train/
    export THICKNESS=/global/cfs/cdirs/m3562/users/hkim/brain_data/instance_thickness.csv
    python $CT_NVAE_PATH/preprocessing/preprocess_brain.py $CSV_PATH $DCM_PATH images_$IMAGE_ID $THICKNESS $NUM_EXAMPLES
else
    srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_images.py -n $NUM_EXAMPLES --dest images_$IMAGE_ID --type $DATA_TYPE
fi

srun -n $SLURM_NTASKS python $CT_NVAE_PATH/preprocessing/create_sinograms.py --dir images_$IMAGE_ID

python $CT_NVAE_PATH/preprocessing/create_splits.py --src images_$IMAGE_ID --dest dataset_$DATASET_ID --train 0.7 --valid 0.2 --test 0.1 -n $NUM_EXAMPLES

python $CT_NVAE_PATH/preprocessing/create_dataset.py --dir dataset_$DATASET_ID --sparse $NUM_SPARSE_ANGLES --random $RANDOM --ring 0 --pnm 1e3

echo "jobend $(date)";pwd