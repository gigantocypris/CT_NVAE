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


# September 7, 2023; 100 example sweep, more epochs
JOB_ID_array_100ex_2 = ["15117860 15117895 15117943 15117985 15118018 15118036 15118094 15118135 15118155",
                        "15118176 15118235 15118283 15118329 15118369 15118402 15118435 15118469 15118512",
                        "15118554 15118578 15118604 15118633 15118653 15118680 15118706 15118740 15118764",
                        "15118791 15118843 15118876 15118919 15118978 15119019 15119066 15119092 15119128",
                        "15119170 15119198 15119237 15119269 15119323 15119360 15119402 15119442 15119474",
                        ]

# September 7, 2023; 10 example sweep, more epochs
JOB_ID_array_10ex_2 = ["15117693 15117724 15117745 15117787 15117810 15117832 15117864 15117897 15117938",
                       "15117719 15117744 15117773 15117812 15117841 15117880 15117908 15117949 15117984",
                       "15117838 15117883 15117910 15117961 15118019 15118041 15118072 15118131 15118149",
                       "15118183 15118222 15118257 15118295 15118353 15118386 15118422 15118476 15118508",
                       "15119180 15119209 15119245 15119282 15119336 15119374 15119416 15119460 15119497",
                       ]

# September 8, 2023; Uniform sparsity sweep, 100 examples
JOB_ID_array_100ex_uniform = ["15182330 15182341 15182367 15182395 15182424 15182466 15182511 15182547 15182594",
                              "15182658 15182712 15182754 15182794 15182833 15182878 15182917 15182969 15183020",
                              "15183150 15183196 15183242 15183286 15183327 15183386 15183435 15183493 15183550",
                              "15183600 15183649 15183693 15183730 15183773 15183812 15183861 15183905 15183949",
                              "15184150 15184182 15184221 15184263 15184298 15184345 15184398 15184461 15184507",
                             ]

# September 8, 2023; Random different sweep, 100 examples
JOB_ID_array_100ex_random = ["15182344 15182369 15182398 15182421 15182458 15182506 15182536 15182586 15182633",
                             "15182692 15182731 15182766 15182813 15182857 15182901 15182944 15182997 15183045",
                             "15183170 15183214 15183256 15183310 15183350 15183398 15183452 15183494 15183555",
                             "15183609 15183648 15183694 15183738 15183779 15183835 15183884 15183934 15183983",
                             "15184153 15184193 15184249 15184306 15184351 15184407 15184468 15184546 15184598",
                            ]

# September 8, 2023; Random uniform sweep, try 1, 100 examples
JOB_ID_array_100ex_rand_uni_1 = ["15182359 15182392 15182435 15182477 15182525 15182569 15182626 15182660 15182696",
                                 "15182747 15182797 15182845 15182894 15182946 15182992 15183044 15183112 15183149",
                                 "15183216 15183270 15183304 15183347 15183399 15183439 15183481 15183521 15183572",
                                 "15183628 15183673 15183722 15183761 15183804 15183860 15183910 15183954 15183996",
                                 "15184157 15184202 15184258 15184295 15184334 15184406 15184472 15184527 15184583",
                                ]

# September 8, 2023; Random uniform sweep, try 2, 100 examples
JOB_ID_array_100ex_rand_uni_2 = ["15182387 15182436 15182480 15182516 15182557 15182595 15182635 15182664 15182697",
                                 "15182776 15182824 15182872 15182925 15182977 15183037 15183087 15183132 15183174",
                                 "15183246 15183292 15183348 15183404 15183449 15183502 15183562 15183612 15183662",
                                 "15183768 15183811 15183863 15183908 15183957 15184004 15184037 15184055 15184072",
                                 "15184158 15184211 15184270 15184311 15184377 15184445 15184501 15184537 15184588",
                                ]

# September 8, 2023; Random uniform sweep, try 3, 100 examples
JOB_ID_array_100ex_rand_uni_3 = ["15182407 15182442 15182473 15182507 15182548 15182593 15182630 15182661 15182699",
                                 "15182752 15182795 15182834 15182882 15182927 15182974 15183014 15183077 15183122",
                                 "15183206 15183253 15183295 15183340 15183379 15183418 15183467 15183514 15183559",
                                 "15183618 15183665 15183715 15183758 15183796 15183836 15183868 15183915 15183963",
                                 "15184164 15184208 15184243 15184292 15184347 15184420 15184479 15184521 15184566",
                                ]

# September 11, 2023; Random sweep, 1 normalizing flow
JOB_ID_array_100ex_rand_normflow = ["15458967 15458974 15458981 15458988 15458995 15459002 15459009 15459016 15459023",
                                    "15459031 15459038 15459045 15459052 15459059 15459068 15459075 15459082 15459089",
                                    "15459098 15459106 15459113 15459120 15459127 15459134 15459141 15459148 15459155",
                                    "15459163 15459170 15459177 15459184 15459191 15459199 15459207 15459214 15459221",
                                    "15459242 15459249 15459256 15459263 15459270 15459277 15459284 15459293 15459302",
                                   ]

JOB_ID_array = JOB_ID_array_100ex_rand_normflow
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

min_ind_vec = min_ind_nelbo
best_JOB_ID = get_min_err_vec(JOB_ID_array_mat, min_ind_vec)
err_vec_NVAE = get_min_err_vec(all_err_vec_NVAE, min_ind_vec)
err_vec_gridrec = get_min_err_vec(all_err_vec_gridrec, min_ind_vec)
err_vec_tv = get_min_err_vec(all_err_vec_tv, min_ind_vec)
err_vec_sirt = get_min_err_vec(all_err_vec_sirt, min_ind_vec)

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
plt.savefig("angle_sweep_compare" + JOB_ID_array_mat[0,0] + ".png")

print("best_JOB_ID", best_JOB_ID)


