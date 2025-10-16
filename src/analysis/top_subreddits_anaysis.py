from platform import node
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from src.utils.data_utils import load_reddit_graph

LIWC_FEATURE_NAMES = [
    "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", 
    "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", 
    "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", 
    "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", 
    "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", 
    "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", 
    "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", 
    "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", 
    "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", 
    "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", 
    "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu"
]

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


def extract_liwc_features(G, top_subreddits=None):
    """
    Extract LIWC features from top subreddits
    """
    subreddit_liwc_data = []
    
    for node in G.nodes():
        if node not in top_subreddits:
            continue
            
        liwc_features = []

        for _, _, edge_data in G.edges(node, data=True):
            props = edge_data['properties']
            liwc_vals = props[21:85]                # LIWC in indices 21-84 of properties
            liwc_features.append(liwc_vals)
    
        mean_liwc = np.mean(liwc_features, axis=0)
        row = {'subreddit': node}
        row.update(dict(zip(LIWC_FEATURE_NAMES, mean_liwc)))
        subreddit_liwc_data.append(row)

    
    return pd.DataFrame(subreddit_liwc_data)


def main():
    G = load_reddit_graph()
    top_in_degree, top_out_degree = find_highest_node_degrees(G)
    high_degree_edges, high_degree_edges_unique = analyze_edge_patterns(G)
    top_subreddits = [node for node, _ in top_in_degree[:20]] + [node for node, _ in top_out_degree[:20]]
    top_subreddits = list(set(top_subreddits)) 
    liwc_df = extract_liwc_features(G, top_subreddits)

if __name__ == "__main__":
    main()