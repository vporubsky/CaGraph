# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:27:47 2019
Author: Veronica Porubsky
Title: Importing and arranging Calcium Imaging Data
Version: 1.0.0
"""
#%% Load key packages/libraries
import csv
import numpy as np

#%% Functions
def getNumRowsInCSV(csv_filename):
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = sum(1 for row in csv_reader)
        print(row_count)
    return row_count

def getNumColsInCSV(csv_filename):
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            col_count = len(row)
            break
        print(col_count)
    return col_count

def convertCSVToMatrix(csv_filename, npy_filename, save_data = True):
    num_rows = getNumRowsInCSV(csv_filename)
    num_cols = getNumColsInCSV(csv_filename)
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        A = np.zeros((num_rows, num_cols))
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for i in range(len(row)):
                    A[line_count, i] = row[i]
                line_count += 1
            else:
                for i in range(len(row)):
                    if row[i] == '':
                        A[line_count, i] = 0.0
                    else:
                        A[line_count, i] = row[i]
                line_count += 1
    if save_data == True:
        np.save(npy_filename, A)
    return(A)      