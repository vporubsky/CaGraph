

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:01:09 2020

@author: Veronica Porubsky

Title: test functionality
"""

from neuronalNetworkGraph import neuronalNetworkGraph
mouse_1_day_1 = neuronalNetworkGraph('mouse_1_with_treatment_day_1_all_calcium_traces.npy')
mouse_1_day_9 = neuronalNetworkGraph('mouse_1_with_treatment_day_9_all_calcium_traces.npy')

mouse_1_day_1.plotSingleNeuronTimeCourse(5)


mouse_2_day_1 = neuronalNetworkGraph('mouse_2_with_treatment_day_1_all_calcium_traces.npy')
mouse_2_day_9 = neuronalNetworkGraph('mouse_2_with_treatment_day_9_all_calcium_traces.npy')