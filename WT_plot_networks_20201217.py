from neuronal_network_graph import neuronal_network_graph as nng
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
sns.set(style="white")
os.chdir('/Users/veronica_porubsky/Documents/GitHub/DG-anxiety-connectivity-studies/PyCharmProject/updated-analysis/day1_day9_treatment')


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