# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:58:55 2020

@author: Veronica Porubsky
"""
import shutil 
import os

# Directory 
directory = "Figures"
  
# Parent Directory path 
parent_dir = os.getcwd()
  
# Path 
path = os.path.join(parent_dir, directory) 
#os.mkdir(path)

for file in os.listdir():
    if file.endswith('.png') or file.endswith('.jpg'):
        shutil.move(os.path.join(parent_dir, file), os.path.join(path, file))