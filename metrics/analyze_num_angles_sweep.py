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


JOB_ID_array_orig = ["14416819 14416820 14416821 14416822 14416823 14416824 14416825 14416826 14416827", # Regular foam
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

JOB_ID_array = JOB_ID_array_100ex_rand_uni_3
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


