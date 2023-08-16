import matplotlib.pyplot as plt
import numpy as np


epoch = np.arange(0,30000,1)
pnm_fraction = 0.9
pnm_warmup_epochs = 10000
pnm_start = 1e1
pnm = 1e3
steepness = -np.log((1-pnm_fraction)/pnm_fraction)/pnm_warmup_epochs
pnm_implement = (pnm-pnm_start+0.5) /( 1 + np.exp(-steepness*epoch))+pnm_start - 0.5

plt.figure()
plt.plot(epoch, pnm_implement)
plt.savefig('pnm_anneal.png')