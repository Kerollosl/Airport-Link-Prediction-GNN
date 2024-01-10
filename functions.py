import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from node2vec import Node2Vec
import torch
from torch import tensor
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay

def plot_detailed_graph(G, country_labels, emphasize_greater=True, degree_threshold=10, edge_highlight=None, labeled_nodes=True):
      # Draw the graph
      plt.figure(figsize=(12,6))
      pos = nx.get_node_attributes(G, "location")

      #degree_threshold = 10
      if emphasize_greater:
        filtered_nodes = [node for node in G.nodes() if G.degree[node] > degree_threshold]
      else:
        filtered_nodes = [node for node in G.nodes() if G.degree[node] < degree_threshold]
      filtered_edges = [(u, v) for u, v in G.edges() if u in filtered_nodes and v in filtered_nodes]

      filtered_node_degrees = {node: G.degree[node] for node in filtered_nodes}


      #node_size = 150
      node_size = [0.00005 * degree**3 for degree in filtered_node_degrees.values()]
      alpha = 1
      #alpha = [0.003 * G.degree[v] for v in G]
      edge_width = [0.00005 * G[u][v]['distance'] for u, v in G.edges()]

      # Draw only the filtered nodes and edges
      nx.draw_networkx_nodes(G, pos, nodelist=filtered_nodes, node_size=node_size, alpha=alpha, node_color='blue')
      nx.draw_networkx_edges(G, pos, edgelist=filtered_edges, width=edge_width, edge_color='0.5')

      if labeled_nodes:
        if emphasize_greater:
          nx.draw_networkx_labels(G, pos, labels=country_labels, font_size=10, font_color='yellow')
        else:
          nx.draw_networkx_labels(G, pos, labels=country_labels, font_size=10, font_color='black')

      if edge_highlight:
        if emphasize_greater:
          highlight_edges = [(u, v) for u, v in filtered_edges if G[u][v]['distance']>edge_highlight]
        else:
          highlight_edges = [(u, v) for u, v in filtered_edges if G[u][v]['distance']<edge_highlight]
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='r', alpha=1)

      #plt.axis('off')
      plt.tight_layout();
      plt.title("Airport Network Graph")
      plt.show()

def print_metrics(G, reverse, count=15, node=False, betweenness=False, closeness=False, distances=False, total_airport_distances=False, distances_per_route=False):

  if node:
    sorted_node_centralities = sorted(node.items(), key=lambda x: x[1], reverse=reverse)
    print("Node Centralities:")
    for node, centrality in sorted_node_centralities[:count]:
        print(f"{node}: {G.nodes[node]['airport']} ({G.nodes[node]['city']},{G.nodes[node]['country']}): {centrality}")

  if betweenness:
    sorted_betweenness_centralities = sorted(betweenness.items(), key=lambda x: x[1], reverse=reverse)
    print("\nBetweenness Centralities:")
    for node, centrality in sorted_betweenness_centralities[:count]:
        print(f"{node}: {G.nodes[node]['airport']} ({G.nodes[node]['city']},{G.nodes[node]['country']}): {centrality}")

  if closeness:
    sorted_closeness_centralities = sorted(closeness.items(), key=lambda x: x[1], reverse=reverse)
    print("\nCloseness Centralities:" )
    for node, centrality in sorted_closeness_centralities[:count]:
        print(f"{node}: {G.nodes[node]['airport']} ({G.nodes[node]['city']},{G.nodes[node]['country']}): {centrality}")

  if distances:
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['distance'], reverse=reverse)
    longest_flights = [(u, v)for u, v, _ in sorted_edges[:count]]
    print("\nLongest/Shortest Flights in the World")
    for u,v in longest_flights:
      print(f"{G.nodes[u]['airport']}({G.nodes[u]['city']},{G.nodes[u]['country']}) to {G.nodes[v]['airport']}({G.nodes[v]['city']},{G.nodes[v]['country']}): {int(G[u][v]['distance']):,} miles")

  if total_airport_distances:
    # Initialize a dictionary to store the sum of distances for each node
    sum_of_distances = {node: 0 for node in G.nodes}

    # Iterate through each edge and update the sum_of_distances dictionary
    for edge in G.edges(data='distance'):
      source, target, distance = edge
      sum_of_distances[source] += distance
      sum_of_distances[target] += distance
    sorted_totals = sorted(sum_of_distances.items(), key=lambda x: x[1], reverse=reverse)[:count]

    print("\nTotal Flight Distance")
    for u, value in sorted_totals:
      print(f"{G.nodes[u]['airport']}({G.nodes[u]['city']},{G.nodes[u]['country']}) has a total distance of all of it's routes of: {int(value):,} miles")

  if distances_per_route:
    sum_of_distances = {node: 0 for node in G.nodes}
    # Iterate through each edge and update the sum_of_distances dictionary
    for edge in G.edges(data='distance'):
      source, target, distance = edge
      sum_of_distances[source] += distance
      sum_of_distances[target] += distance
    keys_to_delete = [u for u in sum_of_distances.keys() if G.degree[u] == 0]
    for u in keys_to_delete:
      del sum_of_distances[u]
    for u, value in sum_of_distances.items():
      sum_of_distances[u]=value/G.degree(u)
    sorted_totals = sorted(sum_of_distances.items(), key=lambda x: x[1], reverse=reverse)[:count]
    print("\nAverage Distance per Route ")
    for u, value in sorted_totals:
      print(f"{G.nodes[u]['airport']}({G.nodes[u]['city']},{G.nodes[u]['country']}) has an average distance per route of: {int(value):,} miles for {G.degree(u)} routes")

def node2vec_embeddings(G, dimensions=128):
  node_index = list(G.nodes())
  # Use the Node2Vec algorithm to generate embeddings
  node2vec = Node2Vec(G, dimensions=dimensions, walk_length=25, num_walks=100, workers=4)

  # Embed nodes
  model = node2vec.fit(window=10, min_count=1)
  # Access the embeddings for nodes
  node_embeddings = [model.wv[node-1] for node in node_index]
  return node_embeddings

def random_embeddings(G, dimensions=128):
    node_index = list(G.nodes())

    # Generate random embeddings for each node
    node_embeddings = [np.random.rand(dimensions) for _ in node_index]

    return node_embeddings

def make_tensor(embedding_list):
  list_of_tensors = [tensor(embedding) for embedding in embedding_list]
  tensor_of_tensors = torch.stack(list_of_tensors, dim=0)
  return tensor_of_tensors

def make_double_tensor(embedding_list_1, embedding_list_2):
    #tensor_1 = tensor(embedding_list_1)
    #tensor_2 = tensor(embedding_list_2)
    combined_tensor = torch.cat((embedding_list_1, embedding_list_2), dim=1)
    return combined_tensor

def edge_tensor(given_edges, num_features, double_tensor):
  temp_tensor = torch.empty((len(given_edges),num_features))
  #print(len(given_edges))
  for i, (u, v) in enumerate(given_edges):
      temp_tensor[i] = double_tensor[u-1] - double_tensor[v-1]
  return temp_tensor

def train_model(model_choice, train_features, train_labels, test_features, test_labels):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_choice.parameters(), lr=0.01)

    num_epochs = 500

    for epoch in range(num_epochs):
        model_choice.train()
        optimizer.zero_grad()
        output = model_choice(train_features)
        loss = criterion(output.squeeze(), train_labels)
        loss.backward()
        optimizer.step()

        # Evaluate the model on the test set after each epoch
        model_choice.eval()
        #test_loss = 0.0
        with torch.no_grad():
            test_output = model_choice(test_features)
            test_loss = criterion(test_output.squeeze(), test_labels).item()

            #test_loss = batch_test_loss
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.2f}, Test Loss: {test_loss:.2f}')
    torch.save(model_choice.state_dict(), "link_prediction_model.pth")

def link_prediction(nodes_df, departing_from, arriving_at, nn_model, actual, num_features, double_tensor):
  test_tensor = edge_tensor([(departing_from, arriving_at)], num_features, double_tensor)
  airport_one = nodes_df.loc[departing_from, ['airport','city', 'area_code']]
  airport_two = nodes_df.loc[arriving_at, ['airport','city', 'area_code']]
  #print(f"Direct Connection between: {airport_one[0]} {airport_one[1]}, {airport_one[2]} to {airport_two[0]} {airport_two[1]}, {airport_two[2]}")

  with torch.no_grad():
      nn_model.eval()
      pred = torch.sigmoid(nn_model(test_tensor))
  return departing_from, airport_one[0], f"{airport_one[1]}, {airport_one[2]}", arriving_at, airport_two[0], f"{airport_two[1]}, {airport_two[2]}", actual, round(pred.item(), 1)

def plot_cm(predicted, actual, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    print(sum(actual == 0), sum(predicted == 0), sum(actual == 1), sum(predicted == 1))
    ConfusionMatrixDisplay.from_predictions(actual, predicted, ax=ax, normalize=None, cmap='inferno')
    ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    plt.title(model_name)
    plt.show()
