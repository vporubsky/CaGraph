# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:53:07 2020

@author: Veronica Porubsky
"""
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
labels = ['A_1_T','B_1_T', 'A_9_T', 'B_9_T', 'A_1_WT','B_1_WT', 'A_9_WT', 'B_9_WT']
avg_num_hubs_thresh_0_5 = [0.023682117,0.034434805,0.037148677,0.032969572,0.044747483,0.048486926,0.02524524,0.026795227]
stdev_num_hubs_thresh_0_5 = [0.031707698,0.052467903,0.062694407,0.048519067,0.014675611,0.0174796,0.026145574,0.033336737]

plt.errorbar(labels, avg_num_hubs_thresh_0_5, stdev_num_hubs_thresh_0_5, linestyle='None',capsize=3, marker='o')
plt.title('Average number of hubs')
plt.savefig('avg_num_hubs_thresh_0_5.png', dpi = 300)

#%% threshold 0.7
plt.figure(2)
labels = ['A_1_T','B_1_T', 'A_9_T', 'B_9_T', 'A_1_WT','B_1_WT', 'A_9_WT', 'B_9_WT']
avg_num_hubs_thresh_0_7 = [0.00716894,0.012459945,0.001811594,0.003019324,0.021783969,0.028101173,0.024350443,0.022941992]
stdev_num_hubs_thresh_0_7 = [0.012856184,0.025403542,0.004437481,0.007395802,0.02121924,0.029309013,0.025809272,0.025254601]

plt.errorbar(labels, avg_num_hubs_thresh_0_7, stdev_num_hubs_thresh_0_7, linestyle='None',capsize=3, marker='o')
plt.title('Average number of hubs')
plt.savefig('avg_num_hubs_thresh_0_7.png', dpi = 300)

#%% threshold 0.3
plt.figure(3)
labels = ['A_1_T','B_1_T', 'A_9_T', 'B_9_T', 'A_1_WT','B_1_WT', 'A_9_WT', 'B_9_WT']
avg_num_hubs_thresh_0_3 = [0.05343062,0.061717017,0.042715117,0.058761794,0.025588026,0.04068854,0.040793641,0.039795569]
stdev_num_hubs_thresh_0_3 = [0.0436411,0.05406801,0.052992619,0.072352146,0.038624336,0.045019779,0.024670324,0.037659867]

plt.errorbar(labels, avg_num_hubs_thresh_0_3, stdev_num_hubs_thresh_0_3, linestyle='None',capsize=3, marker='o')
plt.title('Average number of hubs')
plt.savefig('avg_num_hubs_thresh_0_3.png', dpi = 300)