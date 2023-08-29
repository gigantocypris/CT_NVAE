export CT_NVAE_PATH=$1
export JOB_ID=$2 
export ORIGINAL_SIZE=$3 
export ALGORITHM=$4 
export DATASET_TYPE=$5
python $CT_NVAE_PATH/metrics/analyze_training_results.py --expr_id $JOB_ID --original_size $ORIGINAL_SIZE --algorithm $ALGORITHM --dataset_type $DATASET_TYPE