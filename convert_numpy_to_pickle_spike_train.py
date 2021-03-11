# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:43:31 2020

@author: Veronica Porubsky

Title: Convert spike train arrays (saved as numpy arrays) to pickle files
"""
import os
import pickle
import numpy as np

filenames = os.listdir()

for file in filenames:
    if file.startswith('spikes_'):
        new_name = file.replace('spikes_', '')
        new_name = new_name.replace('.npy', '_spike_train.pkl')
        f = open(new_name, "wb")
        data = np.load(file, allow_pickle=True)
        pickle.dump(data, f)
        f.close()
        

with open('1055-3_D1_spike_train.pkl', 'rb') as f:
    data = pickle.load(f)

print(np.shape(data))
