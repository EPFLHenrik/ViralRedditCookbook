import networkx as nx
import pandas as pd
import numpy as np

from src.utils.data_utils import load_reddit_graph

def find_highest_node_degrees(G):
    """
    Isolate nodes with highest in-degree and out-degree
    """
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    top_in_degree = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    top_out_degree = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 nodes with highest IN-DEGREE:")
    print("Node (Subreddit)\t\tIn-Degree")
    print("-" * 50)
    for node, degree in top_in_degree[:10]:
        print(f"{node:<30}\t{degree}")
    
    print(f"\nTop 10 nodes with highest OUT-DEGREE:")
    print("Node (Subreddit)\t\tOut-Degree")
    print("-" * 50)
    for node, degree in top_out_degree[:10]:
        print(f"{node:<30}\t{degree}")
    
    return top_in_degree, top_out_degree


def analyze_edge_patterns(G, percentile=90):
    """
    Analyze edges connecting nodes with high degrees
    """

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_degree_thresh = np.percentile(list(in_degrees.values()), percentile)
    out_degree_thresh = np.percentile(list(out_degrees.values()), percentile)

    print(f"In-degree threshold for {percentile}th percentile: {in_degree_thresh}")
    print(f"Out-degree threshold for {percentile}th percentile: {out_degree_thresh}")

    high_degree_edges = []

    for src, tgt, data in G.edges(data=True):
        src_out_degree = out_degrees[src]
        tgt_in_degree = in_degrees[tgt]

        if src_out_degree >= out_degree_thresh and tgt_in_degree >= in_degree_thresh:
            high_degree_edges.append({
                'source': src,
                'target': tgt,
                'source_out_degree': src_out_degree,
                'target_in_degree': tgt_in_degree,
                'sentiment': data.get('sentiment', 'Unknown'),
                'timestamp': data.get('timestamp', 'Unknown')
            })
    
    print(f"\nFound {len(high_degree_edges)} edges connecting high-degree nodes...")
    
    # same style dataframe as high_degree_edges but with only unique (source, target) pairs
    high_degree_edges_unique = []
    
    if high_degree_edges:
        high_degree_edges.sort(key=lambda x: x['source_out_degree'] + x['target_in_degree'], reverse=True)
        seen_pairs = set()
        for edge in high_degree_edges:
            pair = (edge['source'], edge['target'])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                high_degree_edges_unique.append(edge)

        print(f"\nFound {len(high_degree_edges_unique)} unique source-target edges connecting high-degree nodes...")

    high_degree_edges = pd.DataFrame(high_degree_edges)
    high_degree_edges_unique = pd.DataFrame(high_degree_edges_unique)

    print("\n\n\nHigh-degree edges:")
    print(high_degree_edges.head(10))
    print("\n\n\nWithout Duplicates:")
    print(high_degree_edges_unique.head(10))

    return high_degree_edges, high_degree_edges_unique


def main():
    G = load_reddit_graph()
    top_in_degree, top_out_degree = find_highest_node_degrees(G)
    high_degree_edges, high_degree_edges_unique = analyze_edge_patterns(G)
if __name__ == "__main__":
    main()