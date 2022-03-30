# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 05:57:47 2019

@author: Veronica Porubsky

This script does not currently work due to errors installing and importing the 
FastLZeroSpikeInference package
"""
import fast
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

gam = 0.98
y = np.power(gam, np.concatenate([np.arange(100), np.arange(100)]))
plt.plot(y)
fit = fast.estimate_spikes(y, gam, 1, False)