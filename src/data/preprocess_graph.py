import pandas as pd
import networkx as nx
import pickle
import os
from tqdm import tqdm


def preprocess_reddit_graph(load_type='body', body_tsv_file='soc-redditHyperlinks-body.tsv', 
                     title_tsv_file='soc-redditHyperlinks-title.tsv', force_reload=False):
    """
    Load Reddit hyperlink data as a NetworkX graph.
    
    Parameters:
    -----------
    load_type : str
        Type of data to load:
        - 'body': Load only body hyperlinks
        - 'title': Load only title hyperlinks  
        - 'merged': Load and merge both body and title hyperlinks
    body_tsv_file : str
        Filename for body hyperlinks TSV file (will be read from data folder)
    title_tsv_file : str
        Filename for title hyperlinks TSV file (will be read from data folder)
    force_reload : bool
        If True, force reload from TSV even if pickle exists
    
    Returns:
    --------
    networkx.MultiDiGraph
        Directed multigraph where nodes are subreddits and edges are posts
    """
    
    data_folder = '../../data'
    
    if load_type == 'body':
        tsv_file = os.path.join(data_folder, body_tsv_file)
        pickle_file = os.path.join(data_folder, 'reddit_graph_body.pkl')
    elif load_type == 'title':
        tsv_file = os.path.join(data_folder, title_tsv_file)
        pickle_file = os.path.join(data_folder, 'reddit_graph_title.pkl')
    elif load_type == 'merged':
        tsv_file = [os.path.join(data_folder, body_tsv_file), os.path.join(data_folder, title_tsv_file)]
        pickle_file = os.path.join(data_folder, 'reddit_graph_merged.pkl')
    else:
        raise ValueError(f"Invalid load_type: {load_type}. Must be 'body', 'title', or 'merged'")
    
    if not force_reload and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            G = pickle.load(f)
            return G
    
    if load_type == 'merged':
        body_df = pd.read_csv(tsv_file[0], sep='\t')
        title_df = pd.read_csv(tsv_file[1], sep='\t')
        
        body_df['source_type'] = 'body'
        title_df['source_type'] = 'title'
        
        df = pd.concat([body_df, title_df], ignore_index=True)
    else:
        df = pd.read_csv(tsv_file, sep='\t')
    
    G = nx.MultiDiGraph()
    
    all_subreddits = set(df['SOURCE_SUBREDDIT'].unique()) | set(df['TARGET_SUBREDDIT'].unique())
    G.add_nodes_from(all_subreddits)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing hyperlinks"):
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        
        properties = []
        if pd.notna(row['PROPERTIES']):
            properties = [float(x) if x != '' else 0.0 for x in str(row['PROPERTIES']).split(',')]
        else:
            raise ValueError("PROPERTIES column is missing")
        
        edge_attrs = {
            'post_id': row['POST_ID'],
            'timestamp': row['TIMESTAMP'],
            'sentiment': row['LINK_SENTIMENT'],
            'properties': properties
        }
        
        if load_type == 'merged':
            edge_attrs['source_type'] = row['source_type']
        
        G.add_edge(source, target, **edge_attrs)
    
    with open(pickle_file, 'wb') as f:
            pickle.dump(G, f)

    return G


if __name__ == "__main__":
    G_merged = preprocess_reddit_graph(load_type='merged',
                            body_tsv_file='soc-redditHyperlinks-body.tsv',
                            title_tsv_file='soc-redditHyperlinks-title.tsv')

    print("Graph loaded successfully, number of nodes: ", G_merged.number_of_nodes(), "number of edges: ", G_merged.number_of_edges())
