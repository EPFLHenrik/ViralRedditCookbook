import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import os
import pickle
from collections import Counter
import csv

from src.utils.data_utils import load_reddit_graph


NEGATIVE_GRAPH_DIR = os.path.join(os.path.dirname(__file__), 'hate_graphs')



def create_negative_sentiment_graph(G, metric='negative_ratio'):
    """
    Create a new directed weighted graph with different metrics to measure subreddit "hate".
    
    Available metrics:
    - 'negative_ratio': Ratio of negative posts to total posts between subreddits
    - 'negative_frequency': Number of negative posts normalized by source subreddit activity
    - 'relative_negative': Negative posts relative to the subreddit's average negative activity
    - 'hate_intensity': Combination of negative ratio and absolute count
    - 'negative_count': Raw negative post count
    - 'source_total_negative': Total negative posts from source subreddit (to all targets)
    - 'source_activity_weighted': Weighted by source subreddit activity
    - 'target_activity_weighted': Weighted by target subreddit activity
    - 'combined_activity': Combined activity (source * target)
    - 'subreddit_size_ratio': Ratio of source subreddit activity to target subreddit activity
    """
    
    pair_counts = defaultdict(lambda: {'negative': 0, 'total': 0})
    subreddit_activity = defaultdict(lambda: {'negative': 0, 'total': 0})
    
    for source, target, attrs in G.edges(data=True):
        sentiment = attrs['sentiment']
        pair_counts[(source, target)]['total'] += 1
        subreddit_activity[source]['total'] += 1
        
        if sentiment == -1:
            pair_counts[(source, target)]['negative'] += 1
            subreddit_activity[source]['negative'] += 1
    
    negative_G = nx.DiGraph()
    
    for (source, target), counts in pair_counts.items():
        negative_count = counts['negative']
        total_count = counts['total']
        
        if negative_count > 0:
            if metric == 'negative_ratio':
                weight = negative_count / total_count
                
            elif metric == 'negative_frequency':
                source_total = subreddit_activity[source]['total']
                weight = negative_count / source_total if source_total > 0 else 0
                
            elif metric == 'relative_negative':
                source_negative = subreddit_activity[source]['negative']
                source_total = subreddit_activity[source]['total']
                avg_negative_ratio = source_negative / source_total if source_total > 0 else 0
                pair_negative_ratio = negative_count / total_count
                weight = pair_negative_ratio / avg_negative_ratio if avg_negative_ratio > 0 else pair_negative_ratio
                
            elif metric == 'hate_intensity':
                negative_ratio = negative_count / total_count
                intensity = negative_ratio * np.log(1 + negative_count)
                weight = intensity
                
            elif metric == 'negative_count':
                weight = negative_count
                
            elif metric == 'source_total_negative':
                source_negative = subreddit_activity[source]['negative']
                weight = source_negative
                
            elif metric == 'source_activity_weighted':
                source_total = subreddit_activity[source]['total']
                weight = negative_count * np.log(1 + source_total)
                
            elif metric == 'target_activity_weighted':
                target_total = subreddit_activity[target]['total']
                weight = negative_count * np.log(1 + target_total)
                
            elif metric == 'combined_activity':
                source_total = subreddit_activity[source]['total']
                target_total = subreddit_activity[target]['total']
                weight = negative_count * np.log(1 + source_total) * np.log(1 + target_total)
                
            elif metric == 'subreddit_size_ratio':
                source_total = subreddit_activity[source]['total']
                target_total = subreddit_activity[target]['total']
                if target_total > 0:
                    weight = negative_count * (source_total / target_total)
                else:
                    weight = negative_count * source_total
                
            else:
                weight = negative_count
            
            negative_G.add_edge(source, target, 
                              weight=weight, 
                              negative_count=negative_count,
                              total_count=total_count,
                              negative_ratio=negative_count/total_count)
    
    return negative_G

def filter_top_edges(G, n_edges=100):
    """
    Filter the graph to keep only the top N edges by weight.
    """
    edges_with_weights = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    
    filtered_G = nx.DiGraph()
    for u, v, weight in edges_with_weights[:n_edges]:
        if not filtered_G.has_node(u):
            filtered_G.add_node(u)
        if not filtered_G.has_node(v):
            filtered_G.add_node(v)
        filtered_G.add_edge(u, v, weight=weight)
    return filtered_G

def create_and_visualize_graph(G, metric, top_n=100, show=False):
    """
    Create and visualize a graph for a specific metric.
    
    Args:
        G: NetworkX graph to analyze
        metric: Metric to use for graph creation
        show: If True, display the plot. If False, only save to file.
    """
    negative_G = create_negative_sentiment_graph(G, metric=metric)
    
    if negative_G.number_of_edges() == 0:
        return
    
    filtered_G = filter_top_edges(negative_G, n_edges=top_n)
    
    if filtered_G.number_of_edges() == 0:
        return
    
    plt.figure(figsize=(20, 16))
    
    pos = nx.shell_layout(filtered_G)

    edges = filtered_G.edges()
    weights = [filtered_G[u][v]['weight'] for u, v in edges]
    
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5 for w in weights]
    else:
        normalized_weights = [0.5] * len(edges)
    
    node_sizes = [filtered_G.degree(node) * 400 + 600 for node in filtered_G.nodes()]
    nx.draw_networkx_nodes(filtered_G, pos, 
                          node_size=node_sizes,
                          node_color='lightcoral',
                          alpha=0.8)
    
    edges_drawn = nx.draw_networkx_edges(filtered_G, pos,
                                        edge_color=normalized_weights,
                                        edge_cmap=plt.cm.rainbow,
                                        width=3,
                                        alpha=0.8,
                                        arrows=True,
                                        arrowsize=25,
                                        arrowstyle='->')
    
    nx.draw_networkx_labels(filtered_G, pos, 
                           font_size=15, 
                           font_weight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    edge_labels = {}
    for u, v in filtered_G.edges():
        weight = filtered_G[u][v]['weight']
        edge_labels[(u, v)] = f"{weight:.2f}"
    
    if edges_drawn and weights:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, 
                                 norm=plt.Normalize(vmin=min(weights), 
                                                   vmax=max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.8, ax=plt.gca())
        cbar.set_label(f'{metric.replace("_", " ").title()} Score', rotation=270, labelpad=20)
    
    metric_descriptions = {
        'negative_ratio': 'weight = negative_count / total_count',
        'negative_frequency': 'weight = negative_count / source_total',
        'relative_negative': 'weight = negative_count / source_total / avg_negative_ratio',
        'hate_intensity': 'weight = negative_ratio * np.log(1 + negative_count)',
        'negative_count': 'weight = negative_count',
        'source_total_negative': 'weight = source_negative',
        'source_activity_weighted': 'weight = negative_count * np.log(1 + source_total)',
        'target_activity_weighted': 'weight = negative_count * np.log(1 + target_total)',
        'combined_activity': 'weight = negative_count * np.log(1 + source_total) * np.log(1 + target_total)',
        'subreddit_size_ratio': 'weight = negative_count * (source_total / target_total)'
    }
    
    plt.title(f"Top {top_n} Negative ({metric_descriptions[metric]})", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = os.path.join(NEGATIVE_GRAPH_DIR, f'strong_negative_sentiment_{metric}.png')
    plt.savefig(output_file, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    metric_folder = os.path.join(NEGATIVE_GRAPH_DIR, metric)
    os.makedirs(metric_folder, exist_ok=True)
    
    edges_csv_path = os.path.join(metric_folder, 'edges.csv')
    nodes_csv_path = os.path.join(metric_folder, 'nodes.csv')
    
    export_graph_to_csv(filtered_G, edges_csv_path)
    export_node_degrees_to_csv(filtered_G, nodes_csv_path)

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


if __name__ == "__main__":
    os.makedirs(NEGATIVE_GRAPH_DIR, exist_ok=True)
    
    G = load_reddit_graph()
    
    create_and_visualize_graph(G, 'negative_frequency', show=False)
    create_and_visualize_graph(G, 'relative_negative', show=False)
    create_and_visualize_graph(G, 'hate_intensity', show=False)
    create_and_visualize_graph(G, 'negative_count', show=False)
    create_and_visualize_graph(G, 'source_total_negative', show=False)
    create_and_visualize_graph(G, 'source_activity_weighted', show=False)
    create_and_visualize_graph(G, 'target_activity_weighted', show=False)
    create_and_visualize_graph(G, 'combined_activity', show=False)
    create_and_visualize_graph(G, 'subreddit_size_ratio', show=False)
    create_and_visualize_graph(G, 'negative_ratio', show=False)
