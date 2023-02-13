# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:30:42 2020

@author: Veronica Porubsky

Title: Test get hubs

Edited on Tue Oct 20 to test showing the distribution of hub values and selecting a cutoff based
on that distribution, given that we expect to see a power law distribution if hubs are present.

"""
from ca_graph import neuronal_network_graph as nng
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
day5_untreated = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv','1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']

nn = nng(day1_untreated[3])
threshold = 0.3

conA = nn.get_context_A_graph(threshold=threshold)
conB = nn.get_context_B_graph(threshold=threshold)

hubs_A, hits_A = nn.get_context_A_hubs(threshold=threshold)
hubs_B, hits_B = nn.get_context_B_hubs(threshold=threshold)

plt.figure(1)
plt.hist(list(hits_A.values()), bins = 20)
plt.title('Hub value distribution -- Context A')
plt.show()

plt.figure(2)
plt.hist(list(hits_B.values()), bins = 20)
plt.title('Hub value distribution -- Context B')
plt.show()


#%% Also test degree information
plt.figure(3)
iterable_degree_obj_A = nn.get_context_A_degree(threshold=threshold)
degree_list_A = []
[degree_list_A.append(x[1]) for x in iterable_degree_obj_A]
plt.hist(degree_list_A, bins = 20)
plt.title('Degree distribution -- Context A')
plt.show()

plt.figure(4)
iterable_degree_obj_B = nn.get_context_B_degree(threshold=threshold)
degree_list_B = []
[degree_list_B.append(x[1]) for x in iterable_degree_obj_B]
plt.hist(degree_list_B, bins = 20)
plt.title('Degree distribution -- Context B')
plt.show()



#%% identify top 5% of values in each Hub value distribution
med_hits_A = np.median(list(hits_A.values()))
std_hits_A = np.std(list(hits_A.values()))
thresh_A = med_hits_A + 2.5*std_hits_A

med_hits_B = np.median(list(hits_B.values()))
std_hits_B = np.std(list(hits_B.values()))
thresh_B = med_hits_B + 2.5*std_hits_B