# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:36:42 2019

@author: Veronica Porubsky

Title: Test Correlation Metrics
"""

from neuronalNetworkGraph import neuronalNetworkGraph
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np

#%% load neuronal network graph

network_1 = neuronalNetworkGraph('mouse_1_with_treatment_day_1_all_calcium_traces.npy')

#%% test Granger Causality
gc_test_dict = grangercausalitytests(np.transpose(network_1.rawCalciumDynamics[1:3, :]), maxlag = 1)[1][0]
new_dict = gc_dict[1][0]
ssr_ftest_p_val = gc_test_dict['ssr_ftest'][1]
chi_2_tes_p_val = gc_test_dict['ssr_chi2test'][1]