"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
# Import visualization functionality
from visualization import interactive_network

# Todo: figure out how the color palettes work
from dg_graph import DGGraph as nng
FC_DATA_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/'
#%%
# WT Data
subject_1 = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-1_D9_smoothed_calcium_traces.csv']
subject_2 = ['1055-2_D1_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv']
subject_3 = ['1055-3_D1_smoothed_calcium_traces.csv', '1055-3_D9_smoothed_calcium_traces.csv']
subject_4 = ['1055-4_D1_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv']
subject_5 = ['14-0_D1_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']
subject_6 = ['122-1_D1_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv']
subject_7 = ['122-2_D1_smoothed_calcium_traces.csv', '122-2_D9_smoothed_calcium_traces.csv']
subject_8 = ['122-3_D1_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']
subject_9 = ['122-4_D1_smoothed_calcium_traces.csv', '122-4_D9_smoothed_calcium_traces.csv']

WT_data = [subject_1, subject_2, subject_3, subject_4, subject_5, subject_6, subject_7, subject_8, subject_9]

# Th Data
subject_1 = ['348-1_D1_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv']
subject_2 = ['349-2_D1_smoothed_calcium_traces.csv', '349-2_D9_smoothed_calcium_traces.csv']
subject_3 = ['386-2_D1_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv']
subject_4 = ['387-4_D1_smoothed_calcium_traces.csv', '387-4_D9_smoothed_calcium_traces.csv']
subject_5 = ['396-1_D1_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv']
subject_6 = ['396-3_D1_smoothed_calcium_traces.csv', '396-3_D9_smoothed_calcium_traces.csv']
subject_7 = ['2-1_D1_smoothed_calcium_traces.csv', '2-1_D9_smoothed_calcium_traces.csv']
subject_8 = ['2-2_D1_smoothed_calcium_traces.csv', '2-2_D9_smoothed_calcium_traces.csv']
subject_9 = ['2-3_D1_smoothed_calcium_traces.csv', '2-3_D9_smoothed_calcium_traces.csv']

Th_data = [subject_1, subject_2, subject_3, subject_4, subject_5, subject_6, subject_7, subject_8, subject_9]
#%%
day_idx=0
data = Th_data
data = FC_DATA_PATH  + subject_2[day_idx]


# data = WT_data
threshold = 0.3

filename = data
nn = nng(filename)

#%% Key functionality
interactive_network(ca_graph_obj=nn, adjust_size_by='communities')


#%% testing
import networkx
G = nn.get_network_graph()
attributes = ['degree', 'CPR', 'hubs','HITS', 'communities']
label_keys = list(map(str, list(nn.labels)))
attribute_dict = {}
for attribute in attributes:
    if attribute == 'degree':
        # Compute the degree of each node and add attribute
        attribute_dict['degree'] = dict(networkx.degree(G))
    elif attribute == 'HITS':
        # Add HITS attribute
        hub_list, hit_vals = nn.get_hubs(graph=G)
        attribute_dict['HITS'] = hit_vals
    elif attribute == 'hubs':
        # Add hubs attribute
        hub_list, hit_vals = nn.get_hubs(graph=G)
        attribute_dict['hubs'] = {i: 1 if i in list(set(hub_list) & set(label_keys)) else 0 for i in label_keys}
    elif attribute == 'CPR':
        # Add correlated pairs attribute
        corr_pair = nn.get_correlated_pair_ratio(graph=G)
        attribute_dict['CPR'] = {i: j for i, j in zip(label_keys, corr_pair)}
    elif attribute == 'communities':
        # Add communities
        c = list(networkx.algorithms.community.greedy_modularity_communities(G))
        sorted(c)
        community_id = {}
        for i in range(len(c)):
            for j in list(c[i]):
                community_id[j] = i
        attribute_dict['communities'] = community_id
    else:
        raise AttributeError('Invalid attribute key entered.')

adjust_node_size = 5
adjust_size_by = 'hubs'
adjusted_node_size = dict([(node, value + adjust_node_size) for node, value in attribute_dict[adjust_size_by].items()])