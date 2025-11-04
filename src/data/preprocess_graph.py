import pandas as pd
import networkx as nx
import pickle
import os
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def preprocess_reddit_graph(load_type='body', body_tsv_file='soc-redditHyperlinks-body.tsv', 
                     title_tsv_file='soc-redditHyperlinks-title.tsv', embeddings_file='web-redditEmbeddings-subreddits.csv', force_reload=False):
    """
    Load Reddit hyperlink data as a NetworkX graph with embeddings.
    
    Args:
        load_type: Type of data to load. Options: 'body' (body hyperlinks only), 'title' (title hyperlinks only), 'merged' (both body and title hyperlinks)
        body_tsv_file: Filename for body hyperlinks TSV file (will be read from data folder)
        title_tsv_file: Filename for title hyperlinks TSV file (will be read from data folder)
        embeddings_file: Filename for subreddit embeddings CSV file (will be read from data folder)
        force_reload: If True, force reload from TSV even if pickle exists
    
    Returns:
        networkx.MultiDiGraph: Directed multigraph where nodes are subreddits with embeddings and edges are posts
    """

    data_folder = DATA_DIR

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
    
    embeddings_path = os.path.join(data_folder, embeddings_file)
    embeddings_df = pd.read_csv(embeddings_path, header=None)
    
    embeddings_dict = {}
    for _, row in tqdm(embeddings_df.iterrows(), total=len(embeddings_df), desc="Loading embeddings"):
        subreddit_name = row[0]
        embedding_vector = row[1:].values.astype(float)
        embeddings_dict[subreddit_name] = embedding_vector
    
    subreddits_with_embeddings = set(embeddings_dict.keys())
    filtered_subreddits = all_subreddits.intersection(subreddits_with_embeddings)
    
    for subreddit in filtered_subreddits:
        G.add_node(subreddit, embedding=embeddings_dict[subreddit])
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing hyperlinks"):
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        
        if source not in filtered_subreddits or target not in filtered_subreddits:
            continue
        
        properties = []
        if pd.notna(row['PROPERTIES']):
            properties = [float(x) for x in str(row['PROPERTIES']).split(',')]
        else:
            raise ValueError("PROPERTIES column is missing")
        
        edge_attrs = {
            'post_id': row['POST_ID'][:6],
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
                            title_tsv_file='soc-redditHyperlinks-title.tsv',
                            embeddings_file='web-redditEmbeddings-subreddits.csv',
                            force_reload=True)

    print("Graph loaded successfully, number of nodes: ", G_merged.number_of_nodes(), "number of edges: ", G_merged.number_of_edges())
