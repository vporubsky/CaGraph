from neuronal_network_graph import NeuronalNetworkGraph as nng
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
sns.set(style="white")
#os.chdir('/Users/veronica_porubsky/Documents/GitHub/DG-anxiety-connectivity-studies/PyCharmProject/updated-analysis/day1_day9_treatment')

day1_treated = ['2-1_D1_smoothed_calcium_traces.csv', '2-2_D1_smoothed_calcium_traces.csv','2-3_D1_smoothed_calcium_traces.csv', '348-1_D1_smoothed_calcium_traces.csv', '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv', '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv', '396-3_D1_smoothed_calcium_traces.csv']
day9_treated = ['2-1_D9_smoothed_calcium_traces.csv', '2-2_D9_smoothed_calcium_traces.csv','2-3_D9_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv', '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv', '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv', '396-3_D9_smoothed_calcium_traces.csv']


subject_1 = ['348-1_D1_smoothed_calcium_traces.csv','348-1_D9_smoothed_calcium_traces.csv']
subject_2 = ['349-2_D1_smoothed_calcium_traces.csv', '349-2_D9_smoothed_calcium_traces.csv']
subject_3 = ['386-2_D1_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv']
subject_4 = ['387-4_D1_smoothed_calcium_traces.csv','387-4_D9_smoothed_calcium_traces.csv']
subject_5 = ['396-1_D1_smoothed_calcium_traces.csv','396-1_D9_smoothed_calcium_traces.csv']
subject_6 = ['396-3_D1_smoothed_calcium_traces.csv', '396-3_D9_smoothed_calcium_traces.csv']
subject_7 = ['2-1_D1_smoothed_calcium_traces.csv', '2-1_D9_smoothed_calcium_traces.csv']
subject_8 = ['2-2_D1_smoothed_calcium_traces.csv', '2-2_D9_smoothed_calcium_traces.csv']
subject_9 = ['2-3_D1_smoothed_calcium_traces.csv', '2-3_D9_smoothed_calcium_traces.csv']

data = [subject_1, subject_2, subject_3, subject_4, subject_5, subject_6, subject_7, subject_8, subject_9]

#%% All measurements, separating contexts
threshold = 0.3
mouse_id_indices = []
for subject_index in [0,1,2,3,4,5,6,7,8]:
    conA = []
    conB = []
    for mouse_id_index in range(len(data[subject_index])):
        filename = data[subject_index][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if subject_index == 0:
            mouse_id_indices.append(mouse_id.replace('_D1', ''))

        nn = nng(filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        conA.append(nn.get_context_A_graph(threshold=threshold))
        conB.append(nn.get_context_B_graph(threshold=threshold))

    plt.figure(figsize=(10,10))
    plt.subplot(221)
    nx.draw_networkx(conA[0], node_size=50, with_labels=False, node_color = 'r', alpha=0.5); plt.title(mouse_id.strip('_D9') + '_D1_conA')
    plt.subplot(222)
    nx.draw_networkx(conB[0], node_size=50, with_labels=False, alpha=0.5); plt.title(mouse_id.strip('_D9') + '_D1_conB');
    plt.subplot(223)
    nx.draw_networkx(conA[1], node_size=50, with_labels=False, node_color = 'r', alpha=0.5); plt.title(mouse_id.strip('_D9') +'_D9_conA')
    plt.subplot(224)
    nx.draw_networkx(conB[1], node_size=50, with_labels=False, alpha=0.5); plt.title(mouse_id.strip('_D9') + '_D9_conB');
    plt.show()