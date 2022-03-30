from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import os

sns.set(style="white")

# %% Keep network configuration then color nodes depending on context-active

subject_1 = ['1055-1_D1_smoothed_calcium_traces.csv','1055-1_D9_smoothed_calcium_traces.csv']
subject_2 = ['1055-2_D1_smoothed_calcium_traces.csv','1055-2_D9_smoothed_calcium_traces.csv']
subject_3 = ['1055-3_D1_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv']
subject_4 = ['1055-4_D1_smoothed_calcium_traces.csv','1055-4_D9_smoothed_calcium_traces.csv']
subject_5 = ['14-0_D1_smoothed_calcium_traces.csv','14-0_D9_smoothed_calcium_traces.csv']
subject_6 = ['122-1_D1_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv']
subject_7 = ['122-2_D1_smoothed_calcium_traces.csv', '122-2_D9_smoothed_calcium_traces.csv']
subject_8 = ['122-3_D1_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']
subject_9 = ['122-4_D1_smoothed_calcium_traces.csv', '122-4_D9_smoothed_calcium_traces.csv']

data = [subject_1, subject_2, subject_3, subject_4, subject_5, subject_6, subject_7, subject_8, subject_9]

# %% All measurements, separating contexts
threshold = 0.3
filename = subject_1[1]
mouse_id = filename.strip('_smoothed_calcium_traces.csv')

nn = nng(filename)
print(f"Executing analyses for {mouse_id}")
num_neurons = nn.num_neurons

conA = nn.get_context_A_graph(threshold=threshold)
conB = nn.get_context_B_graph(threshold=threshold)

POS = nx.spring_layout(conA)
plt.figure(1)
nx.draw_networkx(conA, pos=POS, node_size=50, with_labels=False, node_color='r', alpha=0.5);
plt.title(mouse_id.strip('_D1') + '_D1_conA')
plt.show()
plt.figure(2)
nx.draw_networkx(conB, pos=POS, node_size=50, with_labels=False, node_color='b', alpha=0.5);
plt.title(mouse_id.strip('_D1') + '_D1_conA')
plt.show()

# %%
con_act = list(np.genfromtxt(os.getcwd() + '/mouse_data_files/' + mouse_id + '_neuron_context_active.csv',
                        delimiter=','))  # load context active designations


def get_context_active_indices(con_act):
    nonspecific_indices = []
    con_A_active_indices = []
    con_B_active_indices = []
    for i in range(len(con_act)):
        if con_act[i] == 0:
            nonspecific_indices.append(i)
        elif con_act[i] == 1:
            con_A_active_indices.append(i)
        elif con_act[i] == 2:
            con_B_active_indices.append(i)
    return nonspecific_indices, con_A_active_indices, con_B_active_indices


def get_node_colors(con_act):
    color_list = []
    for i in range(len(con_act)):
        if con_act[i] == 0:
            color_list.append('grey')
        elif con_act[i] == 1:
            color_list.append('red')
        elif con_act[i] == 2:
            color_list.append('blue')
    return color_list


ns_idx, A_idx, B_idx = get_context_active_indices(con_act)
NODE_COLORS = get_node_colors(con_act)

plt.figure(1, figsize=(15, 8))
plt.suptitle(f'DG functional connectivity for subject: {mouse_id} \n Pearsons r threshold: {threshold}', fontsize=18)
axes_1 = nn.plot_circle_graph_network(corr_mat=nn.con_A_pearsons_correlation_matrix, num_lines=len(conA.edges),
                                      threshold=threshold,
                                      node_color_list=NODE_COLORS, title='context A', subplot=121)
axes_2 = nn.plot_circle_graph_network(corr_mat=nn.con_B_pearsons_correlation_matrix, num_lines=len(conB.edges),
                                      threshold=threshold,
                                      node_color_list=NODE_COLORS, title='context B', subplot=122)
plt.show()
plt.close()


#%%
threshold = 0.3


for idx in [0,1]:
    filename = subject_4[idx]
    mouse_id = filename.strip('_smoothed_calcium_traces.csv')

    nn = nng(filename)
    print(f"Executing analyses for {mouse_id}")
    num_neurons = nn.num_neurons

    conA = nn.get_context_A_graph(threshold=threshold)
    conB = nn.get_context_B_graph(threshold=threshold)

    con_act = list(np.genfromtxt(os.getcwd() + '/mouse_data_files/' + mouse_id + '_neuron_context_active.csv',
                                 delimiter=','))  # load context active designations
    NODE_COLORS = get_node_colors(con_act)

    plt.figure(idx)
    nx.draw_networkx(conA, node_size=50, with_labels=True, node_color=NODE_COLORS, alpha=0.5);
    plt.title(mouse_id + '_conA')
    plt.show()
    plt.figure(2)
    nx.draw_networkx(conB, node_size=50, with_labels=True, node_color=NODE_COLORS, alpha=0.5);
    plt.title(mouse_id + '_conB')
    plt.show()