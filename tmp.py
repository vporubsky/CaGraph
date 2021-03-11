'''Remove duplicate files.'''
import os

remove_files_list = []
for file in os.listdir():
    if file.endswith('.py') and file[-5] == ' ':
        remove_files_list.append(file)
    elif len(file) >= 6 and file[-6] == ' ':
        remove_files_list.append(file)

for file in remove_files_list:
    os.remove(file)
    print(f"Removed {file}.")

remove_files_list = []
for file in os.listdir():
    if file.endswith('.npy'):
        remove_files_list.append(file)

for file in remove_files_list:
    os.remove(file)
    print(f"Removed {file}.")
