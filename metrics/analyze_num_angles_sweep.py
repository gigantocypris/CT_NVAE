import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from metrics.utils import get_err_vec

parser = argparse.ArgumentParser(description='Get command line args')
parser.add_argument('--checkpoint_dir', type=str, default='/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts',
                    help='checkpoint directory')
parser.add_argument('--dataset_type', type=str, default='test',
                    help='dataset type', choices=['train', 'valid', 'test'])
parser.add_argument('--metric', type=str, default='PSNR',
                    help='metric', choices=['MSE', 'SSIM', 'PSNR'])
args = parser.parse_args()

if args.metric == 'MSE':
    metric = 0
elif args.metric == 'SSIM':
    metric = 1
elif args.metric == 'PSNR':
    metric = 2

JOB_ID_array = ["14416819 14416820 14416821 14416822 14416823 14416824 14416825 14416826 14416827", # Regular foam
                # "14418069 14418070 14418071 14418073 14418074 14418075 14418077 14418079 14418080", # Foam ring 0.01 jobs (with removal of ring artifact)
                # "14418293 14418295 14418296 14418298 14418299 14418301 14418302 14418303 14418305", # Foam ring 0.01 jobs (without removal of ring artifact)
                # "14422094 14422096 14422098 14422099 14422100 14422101 14422102 14422103 14422104", # Foam ring 0.1 jobs (with removal of ring artifact)
                "14432325 14432331 14432333 14432336 14432338 14432342 14432344 14432347 14432348", # Regular foam
                "14433032 14433035 14433039 14433040 14433041 14433042 14433043 14433044 14433045", # Regular foam
                "14433055 14433058 14433060 14433061 14433062 14433063 14433064 14433065 14433066", # Regular foam
                "14433108 14433111 14433112 14433113 14433114 14433115 14433117 14433118 14433119", # Regular foam
                "14433149 14433152 14433153 14433155 14433157 14433160 14433161 14433163 14433164", # Regular foam
                "14433184 14433186 14433187 14433189 14433190 14433192 14433193 14433194 14433196", # Regular foam
                ]

description = 'NVAE_' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'gridrec'
all_err_vec_NVAE, max_err_vec_NVAE, median_err_vec_NVAE, min_err_vec_NVAE, mean_err_vec_NVAE = get_err_vec(JOB_ID_array, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)

description = '' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'gridrec'
all_err_vec_gridrec, max_err_vec_gridrec, median_err_vec_gridrec, min_err_vec_gridrec, mean_err_vec_gridrec = get_err_vec(JOB_ID_array, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)

description = '' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'tv'
all_err_vec_tv, max_err_vec_tv, median_err_vec_tv, min_err_vec_tv, mean_err_vec_tv = get_err_vec(JOB_ID_array, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)

description = '' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'sirt'
all_err_vec_sirt, max_err_vec_sirt, median_err_vec_sirt, min_err_vec_sirt, mean_err_vec_sirt = get_err_vec(JOB_ID_array, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)


if args.metric == 'MSE':
    err_vec_NVAE = min_err_vec_NVAE
    err_vec_gridrec = min_err_vec_gridrec
    err_vec_tv = min_err_vec_tv
    err_vec_sirt = min_err_vec_sirt
else:
    err_vec_NVAE = max_err_vec_NVAE
    err_vec_gridrec = max_err_vec_gridrec
    err_vec_tv = max_err_vec_tv
    err_vec_sirt = max_err_vec_sirt



plt.figure()
plt.plot(err_vec_NVAE, label='NVAE')
plt.plot(err_vec_gridrec, label='gridrec')
plt.plot(err_vec_tv, label='tv')
plt.plot(err_vec_sirt, label='sirt')
plt.legend()
plt.savefig("angle_sweep_compare.png")

all_neg_log_p = []
all_nelbo = []
for JOB_ID_subarray in JOB_ID_array:
    neg_log_p = []
    nelbo = []
    for JOB_ID in JOB_ID_subarray:
        neg_log_p_i = np.load(f"{args.checkpoint_dir}/eval-{JOB_ID}/final_neg_log_p_{args.dataset_type}.npy")
        nelbo_i = np.load(f"{args.checkpoint_dir}/eval-{JOB_ID}/final_nelbo_{args.dataset_type}.npy")
        neg_log_p.append(neg_log_p_i)
        nelbo.append(nelbo_i)

