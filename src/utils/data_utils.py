import pickle
import pandas as pd
import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def load_reddit_graph(pickle_file=os.path.join(DATA_FOLDER, 'reddit_graph_merged.pkl')):
    """
    Load the pre-built Reddit combined graph from pickle file.
    
    Args:
        pickle_file: Path to the pickle file containing the graph. Defaults to merged graph.
    
    Returns:
        networkx.MultiDiGraph: The loaded Reddit graph with nodes as subreddits and edges as posts.
    """
    
    with open(pickle_file, 'rb') as f:
        G = pickle.load(f)
    
    return G

def load_reddit_df(pickle_file=os.path.join(DATA_FOLDER, 'reddit_df_merged.pkl')):
    """
    Load the pre-processed Reddit dataframe from pickle file.
    
    Args:
        pickle_file: Path to the pickle file containing the dataframe. Defaults to merged dataframe.
    
    Returns:
        pandas.DataFrame: The loaded Reddit dataframe with post properties and metadata.
    """
    
    with open(pickle_file, 'rb') as f:
        df = pd.read_pickle(f)
    
    return df

def load_enhanced_df(pickle_file=os.path.join(DATA_FOLDER, 'enhanced_reddit_df_finalized.pkl')):
    """
    Load the enhanced dataframe from pickle file.
    
    Args:
        pickle_file: Path to the pickle file containing the enhanced dataframe. Defaults to finalized enhanced dataframe.
    
    Returns:
        pandas.DataFrame: The loaded enhanced dataframe with additional Reddit API data.
    """
    
    with open(pickle_file, 'rb') as f:
        df = pd.read_pickle(f)
    
    return df
