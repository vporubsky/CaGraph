%% Script for looking at matrices, converting data to .csv

load 14-0_cellRegistered.mat
y =  cell_registered_struct.cell_to_index_map;
matched_indices = cell_registered_struct.cell_to_index_map;
filename = '14-0_cellRegistered.csv'
csvwrite(filename, matched_indices);  

%% Export csv files of cell matching indices
myFiles = dir(fullfile(pwd,'*.mat')); %gets all wav files in struct
for k = 1:length(myFiles)
    load(myFiles(k).name);
    matched_indices = cell_registered_struct.cell_to_index_map;
    s1 = erase(myFiles(k).name, '.mat' );
    s2 =  '.csv';
    filename = strcat(s1,s2);
    csvwrite(filename, matched_indices);  
end



