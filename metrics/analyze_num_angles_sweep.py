import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from metrics.utils import get_err_vec, get_nelbo_vec, get_min_err_vec

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

"""
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
"""

# September 3, 2023; 1,000 example sweep
JOB_ID_array_1000ex = ["14844050 14844070 14844090 14844107 14844118 14844129 14844139 14844146 14844153",
                       "14889731 14889738 14889749 14889757 14889764 14889774 14889781 14889788 14889799",
                       "14889912 14889921 14889931 14889942 14889951 14889958 14889968 14889977 14889984",
                       "14890012 14890021 14890031 14890038 14890049 14890057 14890064 14890080 14890093",
                       "14890136 14890155 14890167 14890182 14890196 14890208 14890215 14890229 14890246",
                      ]

# September 5, 2023; 100 example sweep
JOB_ID_array_100ex = ["15025942 15025963 15025999 15026018 15026042 15026076 15026101 15026123 15026139",
                      "15025982 15026017 15026048 15026084 15026109 15026126 15026142 15026157 15026169",
                      "15025086 15025102 15025123 15025145 15025168 15025189 15025215 15025235 15025268",
                      "15025305 15025341 15025375 15025427 15025461 15025485 15025506 15025548 15025609",
                      "15025651 15025666 15025680 15025692 15025707 15025721 15025745 15025759 15025800",
                     ]


# September 5, 2023; 10 example sweep
JOB_ID_array_10ex = ["15024651 15024658 15024674 15024689 15024705 15024719 15024738 15024752 15024766",
                     "15024784 15024798 15024823 15024857 15024887 15024911 15024950 15024986 15025039",
                     "15025076 15025089 15025108 15025128 15025150 15025174 15025194 15025224 15025253",
                     "15025292 15025324 15025355 15025385 15025423 15025451 15025477 15025499 15025521",
                     "15025584 15025624 15025637 15025647 15025658 15025673 15025688 15025703 15025718",
                    ]



# September 7, 2023; 100 example sweep
JOB_ID_array_100ex_2 = ["15117860 15117895 15117943 15117985 15118018 15118036 15118094 15118135 15118155",
                        "15118176 15118235 15118283 15118329 15118369 15118402 15118435 15118469 15118512",
                        "15118554 15118578 15118604 15118633 15118653 15118680 15118706 15118740 15118764",
                        "15118791 15118843 15118876 15118919 15118978 15119019 15119066 15119092 15119128",
                        "15119170 15119198 15119237 15119269 15119323 15119360 15119402 15119442 15119474",
                        ]

# September 7, 2023; 10 example sweep
JOB_ID_array_10ex_2 = ["15117693 15117724 15117745 15117787 15117810 15117832 15117864 15117897 15117938",
                       "15117719 15117744 15117773 15117812 15117841 15117880 15117908 15117949 15117984",
                       "15117838 15117883 15117910 15117961 15118019 15118041 15118072 15118131 15118149",
                       "15118183 15118222 15118257 15118295 15118353 15118386 15118422 15118476 15118508",
                       "15119180 15119209 15119245 15119282 15119336 15119374 15119416 15119460 15119497",
                       ]

JOB_ID_array = JOB_ID_array_100ex_2
JOB_ID_array_mat=[]
for JOB_ID_subarray in JOB_ID_array:
    JOB_ID_subarray = JOB_ID_subarray.split(" ")
    JOB_ID_array_mat.append(JOB_ID_subarray)
JOB_ID_array_mat = np.array(JOB_ID_array_mat)

description = 'NVAE_' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'gridrec'
all_err_vec_NVAE, max_err_vec_NVAE, median_err_vec_NVAE, min_err_vec_NVAE, mean_err_vec_NVAE = get_err_vec(JOB_ID_array_mat, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)

description = '' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'gridrec'
all_err_vec_gridrec, max_err_vec_gridrec, median_err_vec_gridrec, min_err_vec_gridrec, mean_err_vec_gridrec = get_err_vec(JOB_ID_array_mat, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)

description = '' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'tv'
all_err_vec_tv, max_err_vec_tv, median_err_vec_tv, min_err_vec_tv, mean_err_vec_tv = get_err_vec(JOB_ID_array_mat, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)

description = '' # description can be 'NVAE_' or 'ring_' or ''
algorithm = 'sirt'
all_err_vec_sirt, max_err_vec_sirt, median_err_vec_sirt, min_err_vec_sirt, mean_err_vec_sirt = get_err_vec(JOB_ID_array_mat, args.checkpoint_dir, args.dataset_type, metric, description, algorithm=algorithm)

all_neg_log_p, all_nelbo, min_ind_neg_log_p, min_ind_nelbo = get_nelbo_vec(JOB_ID_array_mat, args.checkpoint_dir, args.dataset_type)

best_JOB_ID = get_min_err_vec(JOB_ID_array_mat, min_ind_nelbo)
err_vec_NVAE = get_min_err_vec(all_err_vec_NVAE, min_ind_nelbo)
err_vec_gridrec = get_min_err_vec(all_err_vec_gridrec, min_ind_nelbo)
err_vec_tv = get_min_err_vec(all_err_vec_tv, min_ind_nelbo)
err_vec_sirt = get_min_err_vec(all_err_vec_sirt, min_ind_nelbo)

# Compare to actual min/max and values that we get from min nelbo
if args.metric == 'MSE':
    best_err_vec_NVAE = min_err_vec_NVAE
    best_err_vec_gridrec = min_err_vec_gridrec
    best_err_vec_tv = min_err_vec_tv
    best_err_vec_sirt = min_err_vec_sirt
else:
    best_err_vec_NVAE = max_err_vec_NVAE
    best_err_vec_gridrec = max_err_vec_gridrec
    best_err_vec_tv = max_err_vec_tv
    best_err_vec_sirt = max_err_vec_sirt


plt.figure()
plt.plot(err_vec_NVAE, 'b', label='NVAE')
plt.plot(err_vec_gridrec, 'g', label='gridrec')
plt.plot(err_vec_tv, 'r', label='tv')
plt.plot(err_vec_sirt, 'y', label='sirt')
plt.legend()

plt.plot(all_err_vec_NVAE.T, 'b.')
plt.plot(all_err_vec_gridrec.T, 'g.')
plt.plot(all_err_vec_tv.T, 'r.')
plt.plot(all_err_vec_sirt.T, 'y.')
plt.savefig("angle_sweep_compare.png")

print("best_JOB_ID", best_JOB_ID)

breakpoint()


