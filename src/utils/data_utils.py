import pickle
import pandas as pd
import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def load_reddit_graph(pickle_file=os.path.join(DATA_FOLDER, 'reddit_graph_merged.pkl')):
    """Load the pre-built Reddit combined graph from pickle file."""
    
    with open(pickle_file, 'rb') as f:
        G = pickle.load(f)
    
    return G

def load_reddit_df(pickle_file=os.path.join(DATA_FOLDER, 'reddit_df_merged.pkl')):
    """Load the pre-processed Reddit dataframe from pickle file."""
    
    with open(pickle_file, 'rb') as f:
        df = pd.read_pickle(f)
    
    return df
