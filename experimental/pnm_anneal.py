import matplotlib.pyplot as plt
import numpy as np


epoch = np.arange(0,30000,1)
pnm_fraction = 0.9
pnm_warmup_epochs = 10000
pnm_start = 1e1
pnm = 1e3
steepness = -np.log((1-pnm_fraction)/pnm_fraction)/pnm_warmup_epochs
pnm_implement = (2 / (1 + np.exp(-steepness*epoch)) - 1.0)*(pnm-pnm_start) + pnm_start

plt.figure()
plt.plot(epoch, pnm_implement)
plt.savefig('pnm_anneal.png')