# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:53:07 2020

@author: Veronica Porubsky
"""
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
labels = ['1_T','9_T', '1_WT','9_WT']
avg_num_retained_connections_between_contexts_thresh_0_5 = [0.06264426,0.104189948,0.045599245,0.078001366]
stdev_num_retained_connections_between_contexts_thresh_0_5 = [0.024435621,0.105164125,0.036286082,0.088182064]

plt.errorbar(labels, avg_num_retained_connections_between_contexts_thresh_0_5, stdev_num_retained_connections_between_contexts_thresh_0_5, linestyle='None',capsize=3, marker='o')
plt.title('Average number of retained connections between Context A and B')
plt.savefig('avg_num_retained_connections_thresh_0_5.png', dpi = 300)

