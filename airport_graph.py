from functions import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

##########################################Create Graph from DataFrames##########################################


# Load edges and nodes into dfs
edges_df = pd.read_csv("./air-routes-latest-edges.csv")
nodes_df = pd.read_csv("./air-routes-latest-nodes.csv")


# Remove the first row (index 0) from both DataFrames
edges_df = edges_df.iloc[1:]
nodes_df = nodes_df.iloc[1:]

# Combine Latitude and Longitude
nodes_df["location"] = nodes_df.apply(lambda row: (row["lon:double"],row["lat:double"]), axis=1)


# Filter the nodes_df to include only nodes with "airport" label (as opposed to country)
nodes_df = nodes_df[nodes_df['type:string'] == 'airport']

# Filter out country connection edges (node indices >= 3505)
edges_df = edges_df[(edges_df['~from'] < 3505) & (edges_df['~to'] < 3505)]


# Filter data for more efficient iteration
nodes_df = nodes_df[['~id','country:string', 'desc:string', 'city:string','location']]
edges_df = edges_df[['~from', '~to','dist:int']]
nodes_df.columns = ['id', 'area_code', 'airport', 'city','location']
nodes_df.set_index('id', inplace=True)
edges_df.columns = ['from', 'to', 'distance']

G = nx.Graph()

# Add nodes to the graph with labels
for index, row in nodes_df.iterrows():
    node_id = index
    country_key = row['area_code']
    airport = row['airport']
    city = row['city']
    locat = row['location']
    G.add_node(node_id, country=country_key, airport=airport, city=city, location=locat)


# Add edges to the graph with distance attribute
for _, row in edges_df.iterrows():
    source_node = row['from'].astype(int)
    target_node = row['to'].astype(int)
    edge_distance = row['distance']
    G.add_edge(source_node, target_node, distance=edge_distance)

###################################Plot Graph with Nodes in Geographical Locations###################################
#(Latitude and Longitude)

plt.figure(figsize=(12,6))
pos = nx.get_node_attributes(G, "location")

#labels = nx.get_node_attributes(G, 'country')
node_size = [0.005*G.degree[v]**2 for v in G]
#node_size=50
alpha = [0.003*G.degree[v] for v in G]
edge_width = [0.0015*G[u][v]['distance'] for u,v in G.edges()]

nx.draw_networkx(G, pos, alpha=alpha, with_labels=False, edge_color='0.4', node_color='blue', node_size = node_size, width=0.5)

plt.axis('off')
plt.tight_layout();
plt.title("Airport Network Graph")
plt.show()

degrees = dict(G.degree())

##########################################Choose Params for Graph Display##########################################
emphasize_greater = True

# Sort nodes by degree in descending order and take the top 20
labeled_nodes = 15  # min:0, max:100
degree_threshold = 15  # min:0, max:200
edge_highlight_threshold = 8500  # min:0, max:9000

if labeled_nodes:
    if emphasize_greater:
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:labeled_nodes]
    else:
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[-labeled_nodes:]

# Create a dictionary with node number and "Country" attribute for the top 20 nodes
node_to_country = {node: G.nodes[node]["country"] for node, _ in sorted_degrees}

plot_detailed_graph(G, node_to_country, emphasize_greater=emphasize_greater, degree_threshold=degree_threshold,  edge_highlight=edge_highlight_threshold, labeled_nodes=labeled_nodes)

# Calculate and print node centralities, betweenness centralities, and closeness centralities
node_centralities = nx.degree_centrality(G)
betweenness_centralities = nx.betweenness_centrality(G)
closeness_centralities = nx.closeness_centrality(G)


Top_Ranks = True
How_Many = 15  # 0 to 30
# node_Centralities = False
# betweenness_Centralities = False
# closeness_Centralities = False
Flight_Distances = True
Total_Airport_Distances = True
Average_Airport_Distances_Per_Route = True


print_metrics(G, Top_Ranks, How_Many, node_centralities, betweenness_centralities, closeness_centralities, Flight_Distances, Total_Airport_Distances, Average_Airport_Distances_Per_Route)

Atlanta = 1
Boston = 5
Washington_DC = 7
Washington_DC = 10
Orlando = 15
Houston = 11
NYC = 12
LA = 13
Chicago = 18
choice = Boston
print(f"Destinations FROM {G.nodes[choice]['airport']} ({G.nodes[choice]['city']}, {G.nodes[choice]['country']} ({G.degree(choice)} TOTAL))")
print("=========================================================================")
for neighbor in G.edges(choice):
    airport = G.nodes[neighbor[1]]['airport']
    city = G.nodes[neighbor[1]]['city']
    country = G.nodes[neighbor[1]]['country']
    print(f"{airport} ({city}, {country})")

"""#Node2Vec Embeddings and Embeddings as a Tensor"""

node_embeddings = node2vec_embeddings(G)
#node_embeddings = random_embeddings(G)


#Choose Desired Embeddings and Make a tensor out of them

node2vec_tensor = make_tensor(node_embeddings)
location_tensor = make_tensor(list(nodes_df['location']))
double_tensor = make_double_tensor(node2vec_tensor, location_tensor)
print(f"Features shape: {double_tensor.shape}")

##########################################Getting Positive and Negative Edges###########################################

# Positive edges
dgl_graph = dgl.from_networkx(G)
pos_edges = list(G.edges())
np.random.shuffle(pos_edges)

# Split positive edges for training and testing
test_size = int(len(pos_edges) * 0.1)
train_size = int(len(pos_edges) * 0.9)

train_pos_edges, test_pos_edges = pos_edges[:train_size], pos_edges[-test_size:]

# Negative edges
neg_edges = list(nx.non_edges(G))
np.random.shuffle(neg_edges)

# Split negative edges for training and testing
# Use 3/4 the negative edges so that model is not skewed to negative
train_neg_edges, test_neg_edges = neg_edges[:int(train_size*0.75)], neg_edges[int(-test_size*0.75):]

all_pos_neg_edges = train_pos_edges + train_neg_edges + test_pos_edges + test_neg_edges
all_pos_neg_graph = dgl.graph(all_pos_neg_edges)

####################################################Edge Embeddings###################################################

num_features = double_tensor.shape[1]

train_pos_edge_tensors = edge_tensor(train_pos_edges, num_features, double_tensor)
train_neg_edge_tensors = edge_tensor(train_neg_edges, num_features, double_tensor)
test_pos_edge_tensors = edge_tensor(test_pos_edges, num_features, double_tensor)
test_neg_edge_tensors = edge_tensor(test_neg_edges, num_features, double_tensor)

#################################################Edge Feature df#################################################

# Create dfs for positive and negative edges
train_pos_df = pd.DataFrame(train_pos_edge_tensors.numpy(), columns=[f'feature_{i}' for i in range(num_features)])
train_pos_df['label'] = True

train_neg_df = pd.DataFrame(train_neg_edge_tensors.numpy(), columns=[f'feature_{i}' for i in range(num_features)])
train_neg_df['label'] = False

test_pos_df = pd.DataFrame(test_pos_edge_tensors.numpy(), columns=[f'feature_{i}' for i in range(num_features)])
test_pos_df['label'] = True

test_neg_df = pd.DataFrame(test_neg_edge_tensors.numpy(), columns=[f'feature_{i}' for i in range(num_features)])
test_neg_df['label'] = False

# Concatenate positive and negative edge dfs for training and testing
train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)
test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)

# Shuffle dfs again just to be safe with avoiding skewed prediction
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)

################################################Standard NN Model################################################

# Extract features and labels
train_features = torch.tensor(train_df.drop('label', axis=1).values)
train_labels = torch.tensor(train_df['label'].values, dtype=torch.float32)

test_features = torch.tensor(test_df.drop('label', axis=1).values)
test_labels = torch.tensor(test_df['label'].values, dtype=torch.float32)

class standardNN(nn.Module):
    def __init__(self, in_feats):
        super(standardNN, self).__init__()
        self.linear1 = nn.Linear(in_feats, int(in_feats/2))
        self.linear2 = nn.Linear(int(in_feats/2), int(in_feats/4))
        self.linear3 = nn.Linear(int(in_feats/4), 1)
        self.dropout = nn.Dropout(0.35)

    def forward(self, edge_feats):
        # edge_feats: torch.Tensor (N, in_feats)
        x = F.relu(self.linear1(edge_feats))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        pred = self.linear3(x)
        return pred

# Create model
in_feats = train_features.shape[1]
standard_link_pred = standardNN(in_feats)

##########################################GraphSAGE GNN Model##########################################

#GraphSAGE Model
class GraphSAGENet(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers):
        super(GraphSAGENet, self).__init__()
        self.layers = nn.ModuleList([
            dglnn.SAGEConv(in_feats, hidden_feats, 'mean')
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.35)
        self.prediction = nn.Linear(hidden_feats, 1)

    def forward(self, graph, x):
        for layer in self.layers:
            print(x.shape, graph.num_nodes(), graph.num_edges())
            x = F.relu(layer(graph, x))
            x = self.dropout(x)
        x = self.prediction(x)
        return x


in_feats = train_features.shape[1]
hidden_feats = int(in_feats/2)
num_layers = 3

# Create Model
# GS_model = GraphSAGENet(in_feats,hidden_feats, num_layers)

##########################################Train Model(s)##########################################


train_model(standard_link_pred, train_features, train_labels, test_features, test_labels)
# train_model(GS_model)


results = []
for source, dest in test_pos_edges:
    results.append(link_prediction(nodes_df, source, dest, standard_link_pred, 1, num_features, double_tensor))
for source, dest in test_neg_edges:
    results.append(link_prediction(nodes_df, source, dest, standard_link_pred, 0, num_features, double_tensor))

columns = ['source', 'Airport One', 'City, Country One', 'dest', 'Airport Two', 'City, Country Two', 'Actual', 'Prediction']
predictions_df = pd.DataFrame(results, columns=columns)
predictions_df["Average_Degree"] = predictions_df.apply(lambda row: np.mean([G.degree(row['source']), G.degree(row['dest'])]), axis=1)
predictions_df = predictions_df.sort_values(by=['Average_Degree','Prediction'], ascending=False)

false_preds = predictions_df[predictions_df['Actual'] != predictions_df['Prediction']]
false_preds.to_csv("./pred_vs_actual.csv")

plot_cm(predictions_df['Prediction'].astype(int), predictions_df['Actual'], "Confusion Matrix")

precision, recall, fscore, support = precision_recall_fscore_support(predictions_df['Actual'], predictions_df['Prediction'].astype(int), average='macro', zero_division=1)
accuracy = accuracy_score(predictions_df['Actual'], predictions_df['Prediction'].astype(int))
print(f"Accuracy: {accuracy*100 :.2f}%\nPrecision: {precision*100 :.2f}%\nRecall: {recall*100 :.2f}%\nfscore: {fscore*100 :.2f}%")

potential_new_connections = predictions_df[(predictions_df['Actual'] == 0) & (predictions_df['Prediction'] >= 0.8)]
new_links = list(zip(potential_new_connections['source'], potential_new_connections['dest']))
print(f"{len(potential_new_connections)} Potential New Connections")
potential_new_connections.to_csv("./potential_new.csv")

# Draw graph with potential new edges
plt.figure(figsize=(12, 6))
pos = nx.get_node_attributes(G, "location")
node_size = [0.005*G.degree[v]**2 for v in G]
alpha = [0.003*G.degree[v] for v in G]
edge_width = [0.0015*G[u][v]['distance'] for u, v in G.edges()]

nx.draw_networkx(G, pos, alpha=1, with_labels=False, edge_color='0.5', node_color='blue', node_size = node_size, width=0.1)
nx.draw_networkx_edges(G, pos, edgelist=new_links, edge_color="red", width=1.0)

plt.axis('off')
plt.tight_layout();
plt.title("Airport Graph with Potential new Connections")
plt.show()
