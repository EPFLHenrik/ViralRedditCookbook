import pandas as pd
import networkx as nx
from tqdm import tqdm
from src.utils.data_utils import load_final_df
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def preprocess_final_graph(df, pickle_file=os.path.join(DATA_DIR, 'reddit_graph_final.pkl'), force_reload=False):
    """
    Create a MultiDiGraph from final_df with all edges and their attributes.
    
    Args:
        df: DataFrame with SOURCE_SUBREDDIT, TARGET_SUBREDDIT, and other edge attributes
    
    Returns:
        networkx.MultiDiGraph: Directed multigraph where nodes are subreddits and edges are posts
    """
    if not force_reload and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            G = pickle.load(f)
            return G
    
    G = nx.MultiDiGraph()
    
    all_subreddits = set(df['SOURCE_SUBREDDIT'].unique()) | set(df['TARGET_SUBREDDIT'].unique())
    
    for subreddit in all_subreddits:
        G.add_node(subreddit)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing edges"):
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        
        edge_attrs = row.drop(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']).to_dict()
        
        G.add_edge(source, target, **edge_attrs)
    

    if pickle_file:
        with open(pickle_file, 'wb') as f:
            pickle.dump(G, f)
        print(f"Graph created successfully and saved to {pickle_file}")
   
    return G


if __name__ == "__main__":
    df = load_final_df()
    G = preprocess_final_graph(df, force_reload=True)

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
