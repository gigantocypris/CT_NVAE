import os
import numpy as np

JOB_ID_array = "14416819 14416820 14416821 14416822 14416823 14416824 14416825 14416826 14416827".split(" ")

for JOB_ID in JOB_ID_array:
    os.system(f"python $CT_NVAE_PATH/metrics/analyze_training_results.py --expr_id {JOB_ID} --original_size 128 --algorithm gridrec --dataset_type test")
    os.system(f"python $CT_NVAE_PATH/metrics/analyze_training_results.py --expr_id {JOB_ID} --original_size 128 --algorithm sirt --dataset_type test")
    os.system(f"python $CT_NVAE_PATH/metrics/analyze_training_results.py --expr_id {JOB_ID} --original_size 128 --algorithm tv --dataset_type test")