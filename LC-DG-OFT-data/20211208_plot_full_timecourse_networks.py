"""https://melaniewalsh.github.io/Intro-Cultural-Analytics/Network-Analysis/Making-Network-Viz-with-Bokeh.html"""
from bokeh.io import show, save
from bokeh.models import Range1d, Circle, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8
from bokeh.transform import linear_cmap
import networkx
from ca_graph import DGNetworkGraph as nng
import os

data_198_1 = ['198-1_Saline.csv', '198-1_Prop.csv', '198-1_Praz.csv', '198-1_Quetiapine5mg-kg.csv', '198-1_Quetiapine10mg-kg.csv',
            '198-1_CNOSaline.csv', '198-1_CNOPrazosin.csv', '198-1_CNOQuetiapine5mg-kg.csv']

data_202_4 = ['202-4_Saline.csv', '202-4_Prop.csv', '202-4_Praz.csv', '202-4_Quetiapine5mg-kg.csv', '202-4_Quetiapine10mg-kg.csv',
            '202-4_CNOSaline.csv', '202-4_CNOPrazosin.csv', '202-4_CNOQuetiapine5mg-kg.csv']

labels = ['Saline', 'Prop', 'Praz', 'Quetiapine 5mg/kg', 'Quetiapine 10mg/kg', 'CNO + Saline', 'CNO + Praz', 'CNO + Quetiapine']

ca_data = data_198_1

threshold = 0.3


for idx, subject in enumerate(ca_data):
    filename = ca_data
    mouse_id = filename[0:5]
    nn = nng(filename)
    print(f"Executing analyses for {mouse_id}")
    num_neurons = nn.num_neurons

    full_graph = nn.get_network_graph(threshold=threshold)
    G = full_graph

    # Compute the degree of each node and add attribute
    degrees = dict(networkx.degree(G))
    networkx.set_node_attributes(G, name='degree', values=degrees)

    # Add HITS attribute
    hub_list, hit_vals = nn.get_hubs(graph=G)
    networkx.set_node_attributes(G, name='HITS', values=hit_vals)

    # Add communities
    c = list(networkx.algorithms.community.greedy_modularity_communities(G))
    sorted(c)
    community_id = {}
    for i in range(len(c)):
        for j in list(c[i]):
            community_id[j] = i
    networkx.set_node_attributes(G, name='community', values=community_id)

    # Adjusted node size
    number_to_adjust_by = 5
    adjusted_node_size = dict([(node, degree + number_to_adjust_by) for node, degree in networkx.degree(G)])
    networkx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)

    # Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'community'

    # use palette for context active
    color_palette = Blues8

    # Choose a title!
    title = f"Mouse {mouse_id} under condition: {labels[idx]}"

    # Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [
        ("Neuron", "@index"),
        ("Degree", "@degree"),
        ("Context Active", "@context_activity"),
        ("HITS", "@HITS"),
        ('Hub', "@hubs"),
        ("CPR", "@CPR"),
        ("Community", "@community")
    ]

    # Create a plot — set dimensions, toolbar, and title
    plot = figure(tooltips=HOVER_TOOLTIPS,
                  tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                  x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

    # Create a network graph object
    position = networkx.spring_layout(G)
    network_graph = from_networkx(G, position, scale=10, center=(0, 0))

    # Set node sizes and colors according to node degree (color as spectrum of color palette)
    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute,
                                               fill_color=linear_cmap(color_by_this_attribute, color_palette,
                                                                      minimum_value_color, maximum_value_color))

    # Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

    plot.renderers.append(network_graph)
    show(plot)
    # Uncomment line below to save plots
    save(plot, filename= os.getcwd() + f"/visualize_networks/{subject[:-4]}.html")
