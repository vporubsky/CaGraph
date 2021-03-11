# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:09:08 2020

@author: Veronica Porubsky
"""
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
labels = ['A_1_T','B_1_T', 'A_9_T', 'B_9_T', 'A_1_WT','B_1_WT', 'A_9_WT', 'B_9_WT']
avg_num_connected_subnetworks_thresh_0_5 = [0.056168753,0.057081277,0.073845766,0.085033679,0.119452859,0.134905157,0.10733866,0.137415862]
stdev_num_connected_subnetworks_thresh_0_5 = [0.038868684,0.04556699,0.04431875,0.064365781,0.026008396,0.049328797,0.039277908,0.030024567]

plt.errorbar(labels, avg_num_connected_subnetworks_thresh_0_5, stdev_num_connected_subnetworks_thresh_0_5, linestyle='None',capsize=3, marker='o')
plt.title('Average number of connected subnetworks')
plt.savefig('avg_num_connected_subnetworks_thresh_0_5.png', dpi = 300)

plt.figure(2)
labels = ['A_1_T','B_1_T', 'A_9_T', 'B_9_T', 'A_1_WT','B_1_WT', 'A_9_WT', 'B_9_WT']
avg_size_connected_subnetworks_thresh_0_5 = [0.039924352,0.052075153,0.055844764,0.091564411,0.039517256,0.036875426,0.031324149,0.034496274]
stdev_size_connected_subnetworks_thresh_0_5 = [0.019649885,0.043497289,0.04142465,0.070374951,0.011719428,0.013330998,0.00913514,0.016161663]

plt.errorbar(labels, avg_size_connected_subnetworks_thresh_0_5, stdev_size_connected_subnetworks_thresh_0_5, linestyle='None',capsize=3, marker='o')
plt.title('Average size of connected subnetworks')
plt.savefig('avg_size_connected_subnetworks_thresh_0_5.png', dpi = 300)

plt.figure(3)
labels = ['A_1_T','B_1_T', 'A_9_T', 'B_9_T', 'A_1_WT','B_1_WT', 'A_9_WT', 'B_9_WT']
avg_num_fully_active_connected_subnetworks_thresh_0_5 = [0.008520257,0.005845828,0.00968599,0.034152252,0.004166667,0.010241495,0.005950792,0.022004022]
stdev_num_fully_active_connected_subnetworks_thresh_0_5 = [0.01506351,0.012988267,0.016524387,0.064233072,0.00931695,0.018659247,0.009038493,0.030693974]

plt.errorbar(labels, avg_num_fully_active_connected_subnetworks_thresh_0_5, stdev_num_fully_active_connected_subnetworks_thresh_0_5, linestyle='None',capsize=3, marker='o')
plt.title('Average number of fully-active connected subnetworks')
plt.savefig('avg_num_fully_active_connected_subnetworks_thresh_0_5.png', dpi = 300)

plt.figure(4)
labels = ['A_1_T','B_1_T', 'A_9_T', 'B_9_T', 'A_1_WT','B_1_WT', 'A_9_WT', 'B_9_WT']
avg_num_context_active_cells_in_connected_subnetworks_thresh_0_5 = [0.050139106,0.008932247,0.063215334,0.034152252,0.055849315,0.010241495,0.062141816,0.022004022]
stdev_num_context_active_cells_in_connected_subnetworks_thresh_0_5 = [0.062870995,0.013511054,0.061324527,0.064233072,0.03880402,0.018659247,0.059095411,0.030693974]

plt.errorbar(labels, avg_num_context_active_cells_in_connected_subnetworks_thresh_0_5, stdev_num_context_active_cells_in_connected_subnetworks_thresh_0_5, linestyle='None',capsize=3, marker='o')
plt.title('Average number context-active cells in connected subnetworks')
plt.savefig('avg_num_context_active_cells_connected_subnetworks_thresh_0_5.png', dpi = 300)