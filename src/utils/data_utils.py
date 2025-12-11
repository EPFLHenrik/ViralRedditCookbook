import pickle
import pandas as pd
import os
import numpy as np

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

NUMERIC_FEATURES = [
    'Num_characters',
    'Num_characters_no_space',
    'Frac_alpha',
    'Frac_digits',
    'Frac_uppercase',
    'Frac_whitespace',
    'Frac_special',
    'Num_words',
    'Num_unique_words',
    'Num_long_words',
    'Avg_word_length',
    'Num_unique_stopwords',
    'Frac_stopwords',
    'Num_sentences',
    'Num_long_sentences',
    'Avg_chars_per_sentence',
    'Avg_words_per_sentence',
    'Automated_readability_index',
    
    'Sentiment_pos_vader',
    'Sentiment_neg_vader',
    'Sentiment_compound_vader',
    
    'LIWC_Funct',
    'LIWC_Pronoun',
    'LIWC_Ppron',
    'LIWC_I',
    'LIWC_We',
    'LIWC_You',
    'LIWC_SheHe',
    'LIWC_They',
    'LIWC_Ipron',
    'LIWC_Article',
    'LIWC_Verbs',
    'LIWC_AuxVb',
    'LIWC_Past',
    'LIWC_Present',
    'LIWC_Future',
    'LIWC_Adverbs',
    'LIWC_Prep',
    'LIWC_Conj',
    'LIWC_Negate',
    'LIWC_Quant',
    'LIWC_Numbers',
    'LIWC_Swear',
    'LIWC_Social',
    'LIWC_Family',
    'LIWC_Friends',
    'LIWC_Humans',
    'LIWC_Affect',
    'LIWC_Posemo',
    'LIWC_Negemo',
    'LIWC_Anx',
    'LIWC_Anger',
    'LIWC_Sad',
    'LIWC_CogMech',
    'LIWC_Insight',
    'LIWC_Cause',
    'LIWC_Discrep',
    'LIWC_Tentat',
    'LIWC_Certain',
    'LIWC_Inhib',
    'LIWC_Incl',
    'LIWC_Excl',
    'LIWC_Percept',
    'LIWC_See',
    'LIWC_Hear',
    'LIWC_Feel',
    'LIWC_Bio',
    'LIWC_Body',
    'LIWC_Health',
    'LIWC_Sexual',
    'LIWC_Ingest',
    'LIWC_Relativ',
    'LIWC_Motion',
    'LIWC_Space',
    'LIWC_Time',
    'LIWC_Work',
    'LIWC_Achiev',
    'LIWC_Leisure',
    'LIWC_Home',
    'LIWC_Money',
    'LIWC_Relig',
    'LIWC_Death',
    'LIWC_Assent',
    'LIWC_Dissent',
    'LIWC_Nonflu',
    'LIWC_Filler',
    
    'LINK_SENTIMENT',
    
    'upvote_ratio',
    'ups',
    'downs',
    'score',
    'num_comments',
    'num_crossposts',
    'subreddit_subscribers',
    'total_awards_received',
    'view_count',
    'num_reports',
    'num_duplicates',
    'gilded',
    'likes',
    'pwls',
    'wls',
]


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
    calculate_rss=True
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

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed')

    numeric_cols_to_convert = [col for col in NUMERIC_FEATURES if col in df.columns]
    if numeric_cols_to_convert:
        df[numeric_cols_to_convert] = df[numeric_cols_to_convert].apply(pd.to_numeric)

    if drop_nan_upvotes:
        df = df.dropna(subset=['ups', 'upvote_ratio'])
    
    if drop_nan_cluster:
        df = df.dropna(subset=['source_cluster_level_1'])
        df = df.dropna(subset=['target_cluster_level_1'])

    if calculate_rss:
        mask = df['upvote_ratio'] > 0
        total_votes = pd.Series(0.0, index=df.index) # assumption: if ups = 0 or upvote_ratio = 0 we assume downs = 0
        downs = pd.Series(0.0, index=df.index)
    
        total_votes.loc[mask] = df.loc[mask, 'ups'] / df.loc[mask, 'upvote_ratio']
        downs.loc[mask] = total_votes.loc[mask] - df.loc[mask, 'ups']
        
        subreddit_subscribers = np.maximum(df["subreddit_subscribers"], 0)
        
        df = df.assign(
            total_votes=total_votes,
            downs=downs,
            subreddit_subscribers=subreddit_subscribers,
            virality_rss=((2*df["ups"] + df["downs"]) / np.sqrt(subreddit_subscribers + 1))
        )

    return df
