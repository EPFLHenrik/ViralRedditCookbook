import networkx as nx
import pandas as pd
import numpy as np

from src.utils.data_utils import load_reddit_graph

def find_highest_node_degrees(G, output = False):
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



def main():
    G = load_reddit_graph()
    find_highest_node_degrees(G)

if __name__ == "__main__":
    main()