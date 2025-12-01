import pickle
import pandas as pd
import os
import numpy as np

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

def load_final_df(
    pickle_file=os.path.join(DATA_FOLDER, 'reddit_df_final_superclusters.pkl'), 
    drop_nan_upvotes=True, 
    drop_nan_cluster=True, 
    calculate_rss=False
):
    """
    Load the final dataframe from pickle file. Containing both upvotes, downvotes and cluster data.
    
    Args:
        pickle_file: Path to the pickle file containing the final dataframe. Defaults to final dataframe.
    
    Returns:
        pandas.DataFrame: The loaded final dataframe with post properties and metadata.
    """
    with open(pickle_file, 'rb') as f:
        df = pd.read_pickle(f)

    if drop_nan_upvotes:
        df = df.dropna(subset=['ups'])
    
    if drop_nan_cluster:
        df = df.dropna(subset=['source_cluster_level_1'])
        df = df.dropna(subset=['target_cluster_level_1'])

    if calculate_rss:
        df["subreddit_subscribers"] = pd.Series(np.maximum(df["subreddit_subscribers"], 0), index=df.index)
        df["virality_rss"] = df["score"] / (df["subreddit_subscribers"]+1)

    return df
