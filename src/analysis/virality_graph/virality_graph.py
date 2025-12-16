import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict
from src.utils.data_utils import load_final_graph


VIRALITY_GRAPH_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'virality_graphs')


def create_virality_graph(G, aggregation='geometric'):
    """
    Create a new directed weighted graph by aggregating edges between subreddits.
    Aggregates virality_rss values using either geometric mean or arithmetic mean.
    
    Args:
        G: NetworkX MultiDiGraph with edges containing 'virality_rss' attribute
        aggregation: Type of aggregation to use. Options: 'geometric' (default) or 'arithmetic'
    
    Returns:
        networkx.DiGraph: Directed graph where edges between two nodes are aggregated into a single edge
                          with weight = aggregated mean of virality_rss values
    """
    pair_virality = defaultdict(list)
    
    for source, target, attrs in G.edges(data=True):
        virality_rss = attrs['virality_rss']
        if aggregation == 'geometric':
            if virality_rss > 0:
                pair_virality[(source, target)].append(virality_rss)
        else:
            pair_virality[(source, target)].append(virality_rss)
    
    virality_G = nx.DiGraph()
    
    for (source, target), virality_values in pair_virality.items():
        if len(virality_values) > 0:
            if aggregation == 'geometric':
                log_values = np.log(virality_values)
                aggregated_value = np.exp(np.mean(log_values))
            else:
                aggregated_value = np.mean(virality_values)
            
            virality_G.add_edge(source, target,
                              weight=aggregated_value,
                              count=len(virality_values),
                              aggregation=aggregation)
    
    return virality_G


def create_unweighted_virality_graph(virality_G, percentile=75):
    """
    Create an unweighted graph from a weighted virality graph, keeping only edges
    whose weight is in the top percentile of all edge weights.
    
    Args:
        virality_G: NetworkX DiGraph with weighted edges (virality_rss values)
        percentile: Percentile threshold (default: 75, meaning top 25%)
    
    Returns:
        networkx.DiGraph: Unweighted directed graph containing only edges in the top percentile
    """
    if virality_G.number_of_edges() == 0:
        return nx.DiGraph()
    
    weights = [virality_G[u][v]['weight'] for u, v in virality_G.edges()]
    
    if not weights:
        return nx.DiGraph()
    
    threshold = np.percentile(weights, percentile)
    
    unweighted_G = nx.DiGraph()
    
    unweighted_G.add_nodes_from(virality_G.nodes())
    
    for u, v, data in virality_G.edges(data=True):
        if data['weight'] >= threshold:
            unweighted_G.add_edge(u, v)
    
    return unweighted_G


def export_graph_to_csv(G, filename):
    """
    Export graph data to CSV format with columns: subreddit1, subreddit2, weight
    """
    csv_data = []
    for u, v, data in G.edges(data=True):
        csv_data.append({
            'subreddit1': u,
            'subreddit2': v,
            'weight': data['weight']
        })
    
    csv_data.sort(key=lambda x: x['weight'], reverse=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['subreddit1', 'subreddit2', 'weight']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)


def export_node_degrees_to_csv(G, filename):
    """
    Export node degree information to CSV format with columns: subreddit, indegree, outdegree
    """
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    csv_data = []
    for node in G.nodes():
        csv_data.append({
            'subreddit': node,
            'indegree': in_degrees[node],
            'outdegree': out_degrees[node]
        })
    
    csv_data.sort(key=lambda x: (x['outdegree'], x['indegree']), reverse=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['subreddit', 'indegree', 'outdegree']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)


def calculate_virality_metrics(virality_G):
    """
    Calculate weighted PageRank for measuring virality centrality of subreddits.
    
    What weighted PageRank measures here:
    - Edge weights represent aggregated virality_rss values (average impact of posts 
      flowing from source to target subreddit, normalized by subreddit size)
    - Weighted PageRank computes centrality where: a subreddit gets high PageRank if:
      1. It receives links from subreddits that send highly viral content to it
      2. Those source subreddits themselves have high PageRank (viral content flows 
         through influential subreddits)
    - This identifies "virality hubs": subreddits that are central in the network 
      of viral content flow, where viral content is defined by posts that perform 
      exceptionally well relative to their subreddit's size.
    
    In essence, this measures which subreddits are most central in the viral content 
    propagation network, weighted by how viral the content actually is (not just 
    connectivity).
    
    Prints top 10 nodes with highest PageRank scores.
    
    Args:
        virality_G: NetworkX DiGraph with weighted edges (virality_rss values)
    
    Returns:
        dict: Dictionary mapping subreddit names to PageRank scores
    """
    print("Calculating PageRank...")
    pagerank = nx.pagerank(virality_G, weight='weight')
    
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_pagerank[:10]
    
    print("\nTop 10 subreddits by PageRank score:")
    print("-" * 50)
    for i, (subreddit, score) in enumerate(top_10, 1):
        print(f"{i:2d}. {subreddit:30s} {score:.6f}")
    print("-" * 50)
    
    return pagerank



def plot_page_centrality(virality_G, pagerank, n_nodes=20, output_file=None, show=True, export_csv=False):
    """
    Plot a subgraph around the top node by weighted PageRank centrality.
    Uses BFS to explore the neighborhood and collect N nodes.
    Node sizes are proportional to PageRank centrality (virality hub importance).
    
    The PageRank scores represent virality centrality: how central each subreddit is 
    in the network of viral content flow, weighted by the actual virality (RSS) of 
    the content flowing through the edges.
    
    Args:
        virality_G: NetworkX DiGraph with weighted edges (virality_rss values)
        pagerank: Dictionary mapping subreddit names to PageRank scores (virality centrality)
        n_nodes: Number of nodes to include in the subgraph (default: 20)
        output_file: Path to save the plot. If None, saves to default location
        show: Whether to display the plot (default: True)
        export_csv: If True, export the subgraph to CSV files (default: False)
    """
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    top_node = sorted_pagerank[0][0]
    
    neighborhood_nodes = [top_node]
    visited = {top_node}
    queue = [top_node]
    
    while queue and len(neighborhood_nodes) < n_nodes:
        current_node = queue.pop(0)
        
        neighbors = list(virality_G.successors(current_node)) + list(virality_G.predecessors(current_node))
        
        for neighbor in neighbors:
            if neighbor not in visited and len(neighborhood_nodes) < n_nodes:
                visited.add(neighbor)
                neighborhood_nodes.append(neighbor)
                queue.append(neighbor)
    
    subgraph = virality_G.subgraph(neighborhood_nodes).copy()
    
    print(f"\nSubgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    if subgraph.number_of_nodes() == 0:
        print("No nodes in subgraph to plot")
        return
    
    plt.figure(figsize=(20, 16))
    
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    node_colors = []
    node_sizes = []
    max_pagerank = max(pagerank[node] for node in subgraph.nodes())
    min_pagerank = min(pagerank[node] for node in subgraph.nodes())
    
    for node in subgraph.nodes():
        node_colors.append('lightblue')
        normalized_centrality = (pagerank[node] - min_pagerank) / (max_pagerank - min_pagerank)
        node_sizes.append(1000 + normalized_centrality * 3500)
    
    nx.draw_networkx_nodes(subgraph, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8)
    
    edges = subgraph.edges()
    weights = [subgraph[u][v]['weight'] for u, v in edges]
    
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) 
                             if max_weight > min_weight else 0.5 for w in weights]
    else:
        normalized_weights = [0.5] * len(edges)
    
    edges_drawn = nx.draw_networkx_edges(subgraph, pos,
                                        edge_color=normalized_weights,
                                        edge_cmap=plt.cm.viridis,
                                        width=2,
                                        alpha=0.6,
                                        arrows=True,
                                        arrowsize=20,
                                        arrowstyle='->',
                                        node_size=node_sizes)
    
    nx.draw_networkx_labels(subgraph, pos,
                           font_size=15,
                           font_weight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    if edges_drawn and weights:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                 norm=plt.Normalize(vmin=min(weights),
                                                   vmax=max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.8, ax=plt.gca())
        cbar.set_label('Virality RSS Weight', rotation=270, labelpad=20)
    
    plt.title(f"BFS Neighborhood of Top PageRank Node: {top_node} (N={n_nodes})", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.close()
    

    if export_csv:
        os.makedirs(VIRALITY_GRAPH_DIR, exist_ok=True)
        csv_folder = os.path.join(VIRALITY_GRAPH_DIR, f'page_centrality_neighborhood_n{n_nodes}')
        os.makedirs(csv_folder, exist_ok=True)
        
        edges_csv_path = os.path.join(csv_folder, 'edges.csv')
        nodes_csv_path = os.path.join(csv_folder, 'nodes.csv')
        
        export_graph_to_csv(subgraph, edges_csv_path)
        export_node_degrees_to_csv(subgraph, nodes_csv_path)



if __name__ == "__main__":
    G = load_final_graph()
    print(f"Original graph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    virality_G = create_virality_graph(G)
    print(f"Virality graph - Nodes: {virality_G.number_of_nodes()}, Edges: {virality_G.number_of_edges()}")
        
    pagerank = calculate_virality_metrics(virality_G)

    plot_page_centrality(virality_G, show=True, export_csv=True)

