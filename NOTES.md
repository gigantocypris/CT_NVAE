Start with:
module load python
conda activate tomopy

# Installing MPI in the conda environment `tomopy`
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

export NUM_TRAIN=10
export NUM_VAL=10
export NERSC_CPU_ALLOCATION=m2859
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
sbatch -A $NERSC_CPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/create_foam_images_slurm.sh $NUM_TRAIN $NUM_VAL $CT_NVAE_PATH
Submitted batch job 10799095
DONE

export NERSC_GPU_ALLOCATION=m2859_g
sbatch -A $NERSC_GPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/images_to_dataset_slurm.sh $NUM_TRAIN train $CT_NVAE_PATH
Submitted batch job 10800187
DONE

sbatch -A $NERSC_GPU_ALLOCATION -t 00:02:00 $CT_NVAE_PATH/scripts/images_to_dataset_slurm.sh $NUM_VAL valid $CT_NVAE_PATH
Submitted batch job 10800191
DONE

Stitch the dataset:
cd $SCRATCH/output_CT_NVAE
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
python $CT_NVAE_PATH/stitch_dist_datasets.py --num_ranks 4 --dataset_type train
python $CT_NVAE_PATH/stitch_dist_datasets.py --num_ranks 4 --dataset_type valid