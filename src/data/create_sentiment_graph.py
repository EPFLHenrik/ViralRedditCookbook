import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import csv

from src.utils.data_utils import load_reddit_graph


SENTIMENT_GRAPH_DIR = os.path.join(os.path.dirname(__file__), 'sentiment_graphs')


def create_sentiment_graph(G, metric='sentiment_ratio', sentiment='neg'):
    """
    Create a new directed weighted graph with different metrics to measure subreddit sentiment.
    
    Args:
        G: NetworkX graph to analyze
        metric: Metric to use for graph creation
        sentiment: Sentiment type to analyze ('neg' for negative sentiment -1, 'neu_pos' for neutral/positive sentiment 1)
    
    Available metrics:
    - 'sentiment_ratio': Ratio of sentiment posts to total posts between subreddits
    - 'sentiment_frequency': Number of sentiment posts normalized by source subreddit activity
    - 'relative_sentiment': Sentiment posts relative to the subreddit's average sentiment activity
    - 'sentiment_intensity': Combination of sentiment ratio and absolute count
    - 'sentiment_count': Raw sentiment post count
    - 'source_total_sentiment': Total sentiment posts from source subreddit (to all targets)
    - 'source_activity_weighted': Weighted by source subreddit activity
    - 'target_activity_weighted': Weighted by target subreddit activity
    - 'combined_activity': Combined activity (source * target)
    - 'subreddit_size_ratio': Ratio of source subreddit activity to target subreddit activity
    """
    
    sentiment_value = -1 if sentiment == 'neg' else 1
    
    pair_counts = defaultdict(lambda: {'sentiment': 0, 'total': 0})
    subreddit_activity = defaultdict(lambda: {'sentiment': 0, 'total': 0})
    
    for source, target, attrs in G.edges(data=True):
        edge_sentiment = attrs['sentiment']
        pair_counts[(source, target)]['total'] += 1
        subreddit_activity[source]['total'] += 1
        
        if edge_sentiment == sentiment_value:
            pair_counts[(source, target)]['sentiment'] += 1
            subreddit_activity[source]['sentiment'] += 1
    
    sentiment_G = nx.DiGraph()
    
    for (source, target), counts in pair_counts.items():
        sentiment_count = counts['sentiment']
        total_count = counts['total']
        
        if sentiment_count > 0:
            if metric == 'sentiment_ratio':
                weight = sentiment_count / total_count
                
            elif metric == 'sentiment_frequency':
                source_total = subreddit_activity[source]['total']
                weight = sentiment_count / source_total if source_total > 0 else 0
                
            elif metric == 'relative_sentiment':
                source_sentiment = subreddit_activity[source]['sentiment']
                source_total = subreddit_activity[source]['total']
                avg_sentiment_ratio = source_sentiment / source_total if source_total > 0 else 0
                pair_sentiment_ratio = sentiment_count / total_count
                weight = pair_sentiment_ratio / avg_sentiment_ratio if avg_sentiment_ratio > 0 else pair_sentiment_ratio
                
            elif metric == 'sentiment_intensity':
                sentiment_ratio = sentiment_count / total_count
                intensity = sentiment_ratio * np.log(1 + sentiment_count)
                weight = intensity
                
            elif metric == 'sentiment_count':
                weight = sentiment_count
                
            elif metric == 'source_total_sentiment':
                source_sentiment = subreddit_activity[source]['sentiment']
                weight = source_sentiment
                
            elif metric == 'source_sentiment_weighted':
                source_total = subreddit_activity[source]['total']
                weight = sentiment_count * np.log(1 + source_total)
                
            elif metric == 'target_sentiment_weighted':
                target_total = subreddit_activity[target]['total']
                weight = sentiment_count * np.log(1 + target_total)
                
            elif metric == 'combined_activity':
                source_total = subreddit_activity[source]['total']
                target_total = subreddit_activity[target]['total']
                weight = sentiment_count * np.log(1 + source_total) * np.log(1 + target_total)
                
            elif metric == 'subreddit_size_ratio':
                source_total = subreddit_activity[source]['total']
                target_total = subreddit_activity[target]['total']
                if target_total > 0:
                    weight = sentiment_count * (source_total / target_total)
                else:
                    weight = sentiment_count * source_total
                
            else:
                weight = sentiment_count
            
            sentiment_G.add_edge(source, target, 
                              weight=weight, 
                              sentiment_count=sentiment_count,
                              total_count=total_count,
                              sentiment_ratio=sentiment_count/total_count)
    
    return sentiment_G

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

def create_and_visualize_graph(G, metric, sentiment='neg', top_n=100, show=False):
    """
    Create and visualize a graph for a specific metric and sentiment.
    
    Args:
        G: NetworkX graph to analyze
        metric: Metric to use for graph creation
        sentiment: Sentiment type to analyze ('neg' for negative sentiment -1, 'neu_pos' for neutral/positive sentiment 1)
        top_n: Number of top edges to keep for visualization
        show: If True, display the plot. If False, only save to file.
    """
    sentiment_G = create_sentiment_graph(G, metric=metric, sentiment=sentiment)
    
    if sentiment_G.number_of_edges() == 0:
        return
    
    filtered_G = filter_top_edges(sentiment_G, n_edges=top_n)
    
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
    
    if sentiment == 'neg':
        node_color = 'lightcoral'
        edge_cmap = plt.cm.autumn
    else:
        node_color = 'lightgreen'
        edge_cmap = plt.cm.summer
    
    node_sizes = [filtered_G.degree(node) * 400 + 600 for node in filtered_G.nodes()]
    nx.draw_networkx_nodes(filtered_G, pos, 
                          node_size=node_sizes,
                          node_color=node_color,
                          alpha=0.8)
    
    edges_drawn = nx.draw_networkx_edges(filtered_G, pos,
                                        edge_color=normalized_weights,
                                        edge_cmap=edge_cmap,
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
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, 
                                 norm=plt.Normalize(vmin=min(weights), 
                                                   vmax=max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.8, ax=plt.gca())
        cbar.set_label(f'{metric.replace("_", " ").title()} Score', rotation=270, labelpad=20)
    
    metric_descriptions = {
        'sentiment_ratio': 'weight = sentiment_count / total_count',
        'sentiment_frequency': 'weight = sentiment_count / source_total',
        'relative_sentiment': 'weight = sentiment_count / source_total / avg_sentiment_ratio',
        'sentiment_intensity': 'weight = sentiment_ratio * np.log(1 + sentiment_count)',
        'sentiment_count': 'weight = sentiment_count',
        'source_total_sentiment': 'weight = source_sentiment',
        'source_activity_weighted': 'weight = sentiment_count * np.log(1 + source_total)',
        'target_activity_weighted': 'weight = sentiment_count * np.log(1 + target_total)',
        'combined_activity': 'weight = sentiment_count * np.log(1 + source_total) * np.log(1 + target_total)',
        'subreddit_size_ratio': 'weight = sentiment_count * (source_total / target_total)'
    }
    
    sentiment_label = "Negative" if sentiment == 'neg' else "Neutral/Positive"
    plt.title(f"Top {top_n} {sentiment_label} ({metric_descriptions[metric]})", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs(SENTIMENT_GRAPH_DIR, exist_ok=True)
    output_file = os.path.join(SENTIMENT_GRAPH_DIR, f'strong_{sentiment}_sentiment_{metric}.png')
    plt.savefig(output_file, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    metric_folder = os.path.join(SENTIMENT_GRAPH_DIR, f"{sentiment}_{metric}")
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
    os.makedirs(SENTIMENT_GRAPH_DIR, exist_ok=True)
    
    G = load_reddit_graph()
    
    metrics = [
        'sentiment_frequency', 'relative_sentiment', 'sentiment_intensity', 
        'sentiment_count', 'source_total_sentiment', 'source_activity_weighted',
        'target_activity_weighted', 'combined_activity', 'subreddit_size_ratio', 
        'sentiment_ratio'
    ]
    
    for metric in metrics:
        create_and_visualize_graph(G, metric, sentiment='neg', show=False)
    
    for metric in metrics:
        create_and_visualize_graph(G, metric, sentiment='neu_pos', show=False)
