export JOB_ID_ARRAY_ORIG=$1
export INPUT_FILE=$2
export JOB_FINAL_ARRAY=$3
export ORIGINAL_SIZE=$4
export TIME=02:00:00
export DATASET_TYPE=train

conda deactivate
module purge
module load python
export NERSC_GPU_ALLOCATION=m2859_g
conda activate tomopy
export CT_NVAE_PATH=$SCRATCH/CT_NVAE
export WORKING_DIR=$SCRATCH/output_CT_NVAE
export PYTHONPATH=$CT_NVAE_PATH:$PYTHONPATH
mkdir -p $WORKING_DIR
cd $WORKING_DIR


rm -f $INPUT_FILE
echo "INPUT_FILE=$INPUT_FILE"
echo "JOB_ID_ARRAY_ORIG is: "
printf "%s " "${JOB_ID_ARRAY_ORIG[@]}"
echo

for JOB_ID in "${JOB_ID_ARRAY_ORIG[@]}"; do
    # Create analyze_sweep_input.txt
    echo "$CT_NVAE_PATH $JOB_ID $ORIGINAL_SIZE gridrec $DATASET_TYPE" >> $INPUT_FILE
    echo "$CT_NVAE_PATH $JOB_ID $ORIGINAL_SIZE sirt $DATASET_TYPE" >> $INPUT_FILE
    echo "$CT_NVAE_PATH $JOB_ID $ORIGINAL_SIZE tv $DATASET_TYPE" >> $INPUT_FILE
done

export JOB_FINAL_ARRAY_FORMATTED=$(IFS=:; echo "${JOB_FINAL_ARRAY[*]}")
sbatch --dependency=afterok:$JOB_FINAL_ARRAY_FORMATTED -A $NERSC_GPU_ALLOCATION --time=$TIME $CT_NVAE_PATH/slurm/analyze_sweep_sbatch.sh $INPUT_FILE
