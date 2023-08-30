import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Get command line args')
parser.add_argument('--checkpoint_dir', type=str, default='/pscratch/sd/v/vidyagan/output_CT_NVAE/checkpts',
                    help='checkpoint directory')
parser.add_argument('--dataset_type', type=str, default='test',
                    help='dataset type', choices=['train', 'valid', 'test'])
parser.add_argument('--epoch', type=int, default=None,
                    help='epoch to evaluate at')
args = parser.parse_args()



JOB_ID_array = [# "14416819 14416820 14416821 14416822 14416823 14416824 14416825 14416826 14416827", # Regular foam
                "14418069 14418070 14418071 14418073 14418074 14418075 14418077 14418079 14418080", # Foam ring 0.01 jobs (with removal of ring artifact)
                "14418293 14418295 14418296 14418298 14418299 14418301 14418302 14418303 14418305", # Foam ring 0.01 jobs (without removal of ring artifact)
                # "14422094 14422096 14422098 14422099 14422100 14422101 14422102 14422103 14422104", # Foam ring 0.1 jobs (with removal of ring artifact)
                # "14432325 14432331 14432333 14432336 14432338 14432342 14432344 14432347 14432348", # Regular foam
                # "14433032 14433035 14433039 14433040 14433041 14433042 14433043 14433044 14433045", # Regular foam
                # "14433055 14433058 14433060 14433061 14433062 14433063 14433064 14433065 14433066", # Regular foam
                # "14433108 14433111 14433112 14433113 14433114 14433115 14433117 14433118 14433119", # Regular foam
                # "14433149 14433152 14433153 14433155 14433157 14433160 14433161 14433163 14433164", # Regular foam
                # "14433184 14433186 14433187 14433189 14433190 14433192 14433193 14433194 14433196", # Regular foam
                ]


metric = 2 # 0: MSE, 1: SSIM, 2: PSNR
plt.figure()


# all_err_vec_gridrec = []
# all_err_vec_tv = []
# all_err_vec_sirt = []

all_err_vec_NVAE = []
for JOB_ID_subarray in JOB_ID_array:
    err_vec_NVAE = []
    JOB_ID_subarray = JOB_ID_subarray.split(" ")
    for JOB_ID in JOB_ID_subarray:
        err_vec_NVAE_i = np.load(f"{args.checkpoint_dir}/eval-{JOB_ID}/err_vec_NVAE_{args.dataset_type}_algo_gridrec.npy")
        # err_vec_gridrec_i = np.load(f"{args.checkpoint_dir}/eval-{JOB_ID}/err_vec_{args.dataset_type}_algo_gridrec.npy")
        err_vec_NVAE.append(np.mean(err_vec_NVAE_i,axis=0))
    err_vec_NVAE = np.stack(err_vec_NVAE, axis=1)
    all_err_vec_NVAE.append(err_vec_NVAE)
    plt.plot(err_vec_NVAE[metric], 'r.')
    plt.plot(err_vec_NVAE[metric], 'r')
plt.savefig("angle_sweep.png")

all_err_vec_NVAE = np.stack(all_err_vec_NVAE, axis=1) # metrics x trials x datasets

max_err_vec_NVAE = np.max(all_err_vec_NVAE[metric], axis=0)
median_err_vec_NVAE = np.median(all_err_vec_NVAE[metric], axis=0)
min_err_vec_NVAE = np.min(all_err_vec_NVAE[metric], axis=0)
mean_err_vec_NVAE = np.mean(all_err_vec_NVAE[metric], axis=0)

plt.figure()
plt.plot(median_err_vec_NVAE)
plt.savefig("angle_sweep_max.png")
breakpoint()
    