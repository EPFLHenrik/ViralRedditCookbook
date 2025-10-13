import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import csv

from src.utils.data_utils import load_reddit_graph


SENTIMENT_GRAPH_DIR = os.path.join(os.path.dirname(__file__), 'sentiment_graphs')


# TODO: Review calculation of positive and negative metrics

METRIC_DESCRIPTIONS = {
    # Basic Sentiment Metrics
    'sentiment_ratio': 'weight = sentiment_count / total_count',
    'sentiment_frequency': 'weight = sentiment_count / source_total',
    'relative_sentiment': 'weight = sentiment_count / source_total / avg_sentiment_ratio',
    'sentiment_intensity': 'weight = sentiment_ratio * log(1 + sentiment_count)',
    'sentiment_count': 'weight = sentiment_count',
    'source_total_sentiment': 'weight = source_sentiment',
    'source_activity_weighted': 'weight = sentiment_count * log(1 + source_total)',
    'target_activity_weighted': 'weight = sentiment_count * log(1 + target_total)',
    'combined_activity': 'weight = sentiment_count * log(1 + source_total) * log(1 + target_total)',
    'subreddit_size_ratio': 'weight = sentiment_count * (source_total / target_total)',
    
    # Hate-focused Metrics (based on linguistic properties)
    'direct_hate_score': 'weight = mean(negemo + anger + swear + negate)',
    'aggressive_communication': 'weight = mean(uppercase + special_chars + certain + vader_negative)',
    'emotional_toxicity': 'weight = mean(vader_negative + negemo + anger + anxiety + sad + swear)',
    'social_conflict': 'weight = mean(you + they + excl + we_they_ratio)',
    'comprehensive_hate': 'weight = mean(multi_dimensional_hate_score)',
    'hate_intensity_weighted': 'weight = mean(comprehensive_hate * post_length_weight)',
    
    # Positive-focused Metrics (based on linguistic properties)
    'positive_emotion': 'weight = mean(posemo + vader_positive + positive_compound)',
    'social_connection': 'weight = mean(social + family + friends + humans)',
    'positive_activity': 'weight = mean(achiev + leisure + assent)',
    'inclusive_language': 'weight = mean(we_pronoun + inclusive_words)',
    'comprehensive_positive': 'weight = mean(multi_dimensional_positive_score)'
}


def calculate_hate_metrics(properties):
    """
    Calculate all hate-related metrics from linguistic properties.
    
    This function computes various hate speech indicators based on LIWC features,
    VADER sentiment, and linguistic patterns without relying on post labels.
    
    Args:
        properties: List of 86 property values from LIWC and VADER analysis
    
    Returns:
        dict: Dictionary containing precomputed hate metrics:
            - 'comprehensive_hate': Multi-dimensional hate score (0.0-1.0+)
            - 'direct_hate_score': Emotional hate indicators (negemo + anger + swear + negate)
            - 'aggressive_communication': Aggressive language patterns (uppercase + special_chars + certainty + vader_negative)
            - 'emotional_toxicity': Negative emotional intensity (vader_negative + negemo + anger + anxiety + sad + swear)
            - 'social_conflict': Social conflict indicators (you + they + excl + we_they_ratio)
            - 'hate_intensity_weighted': Comprehensive hate weighted by post length
    """
    if not properties or len(properties) < 86:
        return {
            'comprehensive_hate': 0.0,
            'direct_hate_score': 0.0,
            'aggressive_communication': 0.0,
            'emotional_toxicity': 0.0,
            'social_conflict': 0.0,
            'hate_intensity_weighted': 0.0
        }
    
    negemo = properties[48]
    anger = properties[51]
    swear = properties[42]
    negate = properties[39]
    anx = properties[50]
    sad = properties[52]
    
    vader_negative = properties[19]
    vader_compound = properties[21]
    
    uppercase_ratio = properties[4]
    special_chars = properties[6]
    certain = properties[58]
    
    you_pronoun = properties[26]
    they_pronoun = properties[28]
    we_pronoun = properties[25]
    excl = properties[61]
    
    char_count = properties[0]
    word_count = properties[7]
    
    emotional_hate = (negemo + anger + swear + negate) / 4.0
    communication_aggression = (uppercase_ratio + special_chars + certain + vader_negative) / 4.0
    emotional_toxicity = (vader_negative + negemo + anger + anx + sad + swear) / 6.0
    
    we_they_ratio = we_pronoun / (they_pronoun + 0.001)
    social_conflict = (you_pronoun + they_pronoun + excl + min(we_they_ratio, 2.0)) / 4.0
    
    direct_hate_score = emotional_hate
    
    comprehensive_hate = (
        emotional_hate * 0.4 +
        communication_aggression * 0.25 +
        social_conflict * 0.2 +
        min(we_they_ratio, 2.0) * 0.1 +
        max(-vader_compound, 0) * 0.05
    )
    
    length_weight = np.log(1 + char_count) * np.log(1 + word_count)
    hate_intensity_weighted = comprehensive_hate * length_weight
    
    return {
        'comprehensive_hate': comprehensive_hate,
        'direct_hate_score': direct_hate_score,
        'aggressive_communication': communication_aggression,
        'emotional_toxicity': emotional_toxicity,
        'social_conflict': social_conflict,
        'hate_intensity_weighted': hate_intensity_weighted
    }


def calculate_positive_metrics(properties):
    """
    Calculate all positive sentiment metrics from linguistic properties.
    
    This function computes various positive sentiment indicators based on LIWC features,
    VADER sentiment, and linguistic patterns without relying on post labels.
    
    Args:
        properties: List of 86 property values from LIWC and VADER analysis
    
    Returns:
        dict: Dictionary containing precomputed positive metrics:
            - 'positive_emotion': Positive emotional indicators (posemo + vader_positive + positive_compound)
            - 'social_connection': Social connection indicators (social + family + friends + humans)
            - 'positive_activity': Positive activity indicators (achiev + leisure + assent)
            - 'inclusive_language': Inclusive language indicators (we_pronoun + inclusive_words)
            - 'comprehensive_positive': Multi-dimensional positive score (0.0-1.0+)
    """
    if not properties or len(properties) < 86:
        return {
            'positive_emotion': 0.0,
            'social_connection': 0.0,
            'positive_activity': 0.0,
            'inclusive_language': 0.0,
            'comprehensive_positive': 0.0
        }
    
    posemo = properties[47]
    vader_positive = properties[18]
    vader_compound = properties[21]
    
    social = properties[43]
    family = properties[44]
    friends = properties[45]
    humans = properties[46]
    
    achiev = properties[67]
    leisure = properties[68]
    assent = properties[75]
    
    we_pronoun = properties[25]
    incl = properties[60]
    
    positive_emotion = (posemo + vader_positive + max(vader_compound, 0)) / 3.0
    social_connection = (social + family + friends + humans) / 4.0
    positive_activity = (achiev + leisure + assent) / 3.0
    inclusive_language = (we_pronoun + incl) / 2.0
    
    comprehensive_positive = (
        positive_emotion * 0.4 +
        social_connection * 0.25 +
        positive_activity * 0.2 +
        inclusive_language * 0.15
    )
    
    return {
        'positive_emotion': positive_emotion,
        'social_connection': social_connection,
        'positive_activity': positive_activity,
        'inclusive_language': inclusive_language,
        'comprehensive_positive': comprehensive_positive
    }


def create_sentiment_graph(G, metric='sentiment_ratio', sentiment='neg'):
    """
    Create a new directed weighted graph with different metrics to measure subreddit sentiment.
    
    Args:
        G: NetworkX graph to analyze
        metric: Metric to use for graph creation
        sentiment: Sentiment type to analyze ('neg' for negative sentiment -1, 'neu_pos' for neutral/positive sentiment 1)
    
    Available metrics:
    
    Basic Sentiment Metrics:
    - 'sentiment_ratio': Ratio of sentiment posts to total posts between subreddits
    - 'sentiment_frequency': Number of sentiment posts normalized by source subreddit activity
    - 'relative_sentiment': Sentiment posts relative to the subreddit's average sentiment activity
    - 'sentiment_intensity': Combination of sentiment ratio and absolute count
    - 'sentiment_count': Raw sentiment post count
    - 'source_total_sentiment': Total sentiment posts from source subreddit (to all targets)
    - 'source_activity_weighted': Weighted by source subreddit activity
    - 'target_activity_weighted': Weighted by target subreddit activity
    - 'combined_activity': Combined activity (source * target)
    - 'subreddit_size_ratio': Ratio of source subreddit activity to target subreddit activity
    
    Hate-focused metrics (based on linguistic properties):
    - 'direct_hate_score': Emotional hate indicators (LIWC_Negemo + Anger + Swear + Negate)
    - 'aggressive_communication': Aggressive language patterns (Uppercase + Special_chars + Certain + VADER_negative)
    - 'emotional_toxicity': Negative emotional intensity (VADER_negative + Negemo + Anger + Anxiety + Sad + Swear)
    - 'social_conflict': Social conflict indicators (You + They + Excl + We/They ratio)
    - 'comprehensive_hate': Multi-dimensional hate score combining all hate indicators
    - 'hate_intensity_weighted': Comprehensive hate score weighted by post length
    
    Positive-focused metrics (based on linguistic properties):
    - 'positive_emotion': Positive emotional indicators (LIWC_Posemo + VADER_positive + positive_compound)
    - 'social_connection': Social connection indicators (Social + Family + Friends + Humans)
    - 'positive_activity': Positive activity indicators (Achiev + Leisure + Assent)
    - 'inclusive_language': Inclusive language indicators (We_pronoun + Inclusive_words)
    - 'comprehensive_positive': Multi-dimensional positive score combining all positive indicators
    """
    
    sentiment_value = -1 if sentiment == 'neg' else 1
    
    pair_counts = defaultdict(lambda: {'sentiment': 0, 'total': 0, 'hate_metrics': [], 'positive_metrics': [], 'properties': []})
    subreddit_activity = defaultdict(lambda: {'sentiment': 0, 'total': 0, 'hate_metrics': [], 'positive_metrics': [], 'properties': []})
    
    for source, target, attrs in G.edges(data=True):
        edge_sentiment = attrs['sentiment']
        post_label = attrs.get('post_label', 0)
        properties = attrs.get('properties', [])

        pair_counts[(source, target)]['total'] += 1
        subreddit_activity[source]['total'] += 1
        pair_counts[(source, target)]['properties'].append(properties)
        subreddit_activity[source]['properties'].append(properties)

        if edge_sentiment == sentiment_value:
            pair_counts[(source, target)]['sentiment'] += 1
            subreddit_activity[source]['sentiment'] += 1

            hate_metrics = calculate_hate_metrics(properties)
            positive_metrics = calculate_positive_metrics(properties)
            
            pair_counts[(source, target)]['hate_metrics'].append(hate_metrics)
            pair_counts[(source, target)]['positive_metrics'].append(positive_metrics)
            subreddit_activity[source]['hate_metrics'].append(hate_metrics)
            subreddit_activity[source]['positive_metrics'].append(positive_metrics)
    
    sentiment_G = nx.DiGraph()
    
    for (source, target), counts in pair_counts.items():
        sentiment_count = counts['sentiment']
        total_count = counts['total']
        
        if sentiment_count > 0:
            if metric == 'sentiment_ratio':
                weight = sentiment_count / total_count
                
            elif metric == 'sentiment_frequency':
                source_total = subreddit_activity[source]['total']
                weight = sentiment_count / source_total if source_total > 0 else 0
                
            elif metric == 'relative_sentiment':
                source_sentiment = subreddit_activity[source]['sentiment']
                source_total = subreddit_activity[source]['total']
                avg_sentiment_ratio = source_sentiment / source_total if source_total > 0 else 0
                pair_sentiment_ratio = sentiment_count / total_count
                weight = pair_sentiment_ratio / avg_sentiment_ratio if avg_sentiment_ratio > 0 else pair_sentiment_ratio
                
            elif metric == 'sentiment_intensity':
                sentiment_ratio = sentiment_count / total_count
                intensity = sentiment_ratio * np.log(1 + sentiment_count)
                weight = intensity
                
            elif metric == 'sentiment_count':
                weight = sentiment_count
                
            elif metric == 'source_total_sentiment':
                source_sentiment = subreddit_activity[source]['sentiment']
                weight = source_sentiment
                
            elif metric == 'source_sentiment_weighted':
                source_total = subreddit_activity[source]['total']
                weight = sentiment_count * np.log(1 + source_total)
                
            elif metric == 'target_sentiment_weighted':
                target_total = subreddit_activity[target]['total']
                weight = sentiment_count * np.log(1 + target_total)
                
            elif metric == 'combined_activity':
                source_total = subreddit_activity[source]['total']
                target_total = subreddit_activity[target]['total']
                weight = sentiment_count * np.log(1 + source_total) * np.log(1 + target_total)
                
            elif metric == 'subreddit_size_ratio':
                source_total = subreddit_activity[source]['total']
                target_total = subreddit_activity[target]['total']
                if target_total > 0:
                    weight = sentiment_count * (source_total / target_total)
                else:
                    weight = sentiment_count * source_total
                
        
            elif metric == 'direct_hate_score':
                hate_metrics_list = counts['hate_metrics']
                if hate_metrics_list:
                    scores = [metrics['direct_hate_score'] for metrics in hate_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'aggressive_communication':
                hate_metrics_list = counts['hate_metrics']
                if hate_metrics_list:
                    scores = [metrics['aggressive_communication'] for metrics in hate_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'emotional_toxicity':
                hate_metrics_list = counts['hate_metrics']
                if hate_metrics_list:
                    scores = [metrics['emotional_toxicity'] for metrics in hate_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'social_conflict':
                hate_metrics_list = counts['hate_metrics']
                if hate_metrics_list:
                    scores = [metrics['social_conflict'] for metrics in hate_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'comprehensive_hate':
                hate_metrics_list = counts['hate_metrics']
                if hate_metrics_list:
                    scores = [metrics['comprehensive_hate'] for metrics in hate_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'hate_intensity_weighted':
                hate_metrics_list = counts['hate_metrics']
                if hate_metrics_list:
                    scores = [metrics['hate_intensity_weighted'] for metrics in hate_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
            
            
            elif metric == 'positive_emotion':
                positive_metrics_list = counts['positive_metrics']
                if positive_metrics_list:
                    scores = [metrics['positive_emotion'] for metrics in positive_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'social_connection':
                positive_metrics_list = counts['positive_metrics']
                if positive_metrics_list:
                    scores = [metrics['social_connection'] for metrics in positive_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'positive_activity':
                positive_metrics_list = counts['positive_metrics']
                if positive_metrics_list:
                    scores = [metrics['positive_activity'] for metrics in positive_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'inclusive_language':
                positive_metrics_list = counts['positive_metrics']
                if positive_metrics_list:
                    scores = [metrics['inclusive_language'] for metrics in positive_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                    
            elif metric == 'comprehensive_positive':
                positive_metrics_list = counts['positive_metrics']
                if positive_metrics_list:
                    scores = [metrics['comprehensive_positive'] for metrics in positive_metrics_list]
                    weight = np.mean(scores)
                else:
                    weight = 0.0
                
            else:
                weight = sentiment_count
            
            sentiment_G.add_edge(source, target, 
                              weight=weight, 
                              sentiment_count=sentiment_count,
                              total_count=total_count,
                              sentiment_ratio=sentiment_count/total_count)
    
    return sentiment_G


def create_combined_graph(G, metric_pos='sentiment_ratio', metric_neg='sentiment_ratio', top_n_neg=50, top_n_pos=50):
    """
    Create a combined graph with both negative and positive sentiment edges.
    
    This function creates a single graph containing both negative and positive sentiment
    edges, allowing for direct comparison between different types of sentiment flows
    between subreddits.
    
    Args:
        G: NetworkX graph to analyze
        metric_pos: Metric to use for positive sentiment edges (see create_sentiment_graph for available metrics)
        metric_neg: Metric to use for negative sentiment edges (see create_sentiment_graph for available metrics)
        top_n_neg: Number of top negative edges to keep (ranked by weight)
        top_n_pos: Number of top positive edges to keep (ranked by weight)
    
    Returns:
        NetworkX DiGraph with both negative and positive edges, where each edge has:
            - 'weight': The calculated metric value
            - 'sentiment': Either 'negative' or 'positive'
            - 'sentiment_count': Number of posts with that sentiment
            - 'total_count': Total number of posts between the subreddits
            - 'sentiment_ratio': Ratio of sentiment posts to total posts
    """
    neg_graph = create_sentiment_graph(G, metric=metric_neg, sentiment='neg')
    pos_graph = create_sentiment_graph(G, metric=metric_pos, sentiment='neu_pos')
    
    filtered_neg = filter_top_edges(neg_graph, n_edges=top_n_neg)
    filtered_pos = filter_top_edges(pos_graph, n_edges=top_n_pos)
    
    combined_G = nx.DiGraph()
    
    for u, v, data in filtered_neg.edges(data=True):
        combined_G.add_edge(u, v, 
                          weight=data['weight'],
                          sentiment='negative',
                          sentiment_count=data.get('sentiment_count', 0),
                          total_count=data.get('total_count', 0),
                          sentiment_ratio=data.get('sentiment_ratio', 0))
    
    for u, v, data in filtered_pos.edges(data=True):
        combined_G.add_edge(u, v, 
                          weight=data['weight'],
                          sentiment='positive',
                          sentiment_count=data.get('sentiment_count', 0),
                          total_count=data.get('total_count', 0),
                          sentiment_ratio=data.get('sentiment_ratio', 0))
    
    return combined_G


def filter_top_edges(G, n_edges=100):
    """
    Filter the graph to keep only the top N edges by weight.
    """
    edges_with_weights = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    
    filtered_G = nx.DiGraph()
    for u, v, weight in edges_with_weights[:n_edges]:
        if not filtered_G.has_node(u):
            filtered_G.add_node(u)
        if not filtered_G.has_node(v):
            filtered_G.add_node(v)
        filtered_G.add_edge(u, v, weight=weight)
    return filtered_G


def create_and_visualize_combined_graph(G, metric_pos='sentiment_ratio', metric_neg='sentiment_ratio', top_n_neg=50, top_n_pos=50, show=False):
    """
    Create and visualize a combined graph with both negative and positive sentiment edges.
    
    This function creates a combined sentiment graph and visualizes it with:
    - Red edges for negative sentiment (using autumn colormap, bottom 75%)
    - Green edges for positive sentiment (using summer colormap, bottom 75%)
    - Separate color scales for each sentiment type
    - Edge weights displayed as numerical labels
    - Centered title with metric descriptions
    
    Args:
        G: NetworkX graph to analyze
        metric_pos: Metric to use for positive sentiment edges (see create_sentiment_graph for available metrics)
        metric_neg: Metric to use for negative sentiment edges (see create_sentiment_graph for available metrics)
        top_n_neg: Number of top negative edges to keep for visualization (ranked by weight)
        top_n_pos: Number of top positive edges to keep for visualization (ranked by weight)
        show: If True, display the plot. If False, only save to file.
    
    Output:
        - Saves PNG file to sentiment_graphs/ directory
        - Exports CSV files with edge and node data
        - Creates subdirectory with metric-specific data
    """
    combined_G = create_combined_graph(G, metric_pos=metric_pos, metric_neg=metric_neg, top_n_neg=top_n_neg, top_n_pos=top_n_pos)
    
    if combined_G.number_of_edges() == 0:
        return
    
    plt.figure(figsize=(20, 12))
    
    pos = nx.shell_layout(combined_G)
    
    neg_edges = [(u, v) for u, v, data in combined_G.edges(data=True) if data.get('sentiment') == 'negative']
    pos_edges = [(u, v) for u, v, data in combined_G.edges(data=True) if data.get('sentiment') == 'positive']
    
    neg_weights = [combined_G[u][v]['weight'] for u, v in neg_edges]
    pos_weights = [combined_G[u][v]['weight'] for u, v in pos_edges]
    
    all_weights = neg_weights + pos_weights
    if all_weights:
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        if max_weight > min_weight:
            neg_normalized = [(w - min_weight) / (max_weight - min_weight) for w in neg_weights]
            pos_normalized = [(w - min_weight) / (max_weight - min_weight) for w in pos_weights]
        else:
            neg_normalized = [0.5] * len(neg_weights)
            pos_normalized = [0.5] * len(pos_weights)
    else:
        neg_normalized = [0.5] * len(neg_weights)
        pos_normalized = [0.5] * len(pos_weights)
    
    node_sizes = [combined_G.degree(node) * 400 + 600 for node in combined_G.nodes()]
    nx.draw_networkx_nodes(combined_G, pos, 
                          node_size=node_sizes,
                          node_color='lightgray',
                          alpha=0.8)
    
    autumn_truncated = plt.cm.autumn(np.linspace(0, 0.75, 256))
    summer_truncated = plt.cm.summer(np.linspace(0, 0.75, 256))
    
    if neg_edges:
        nx.draw_networkx_edges(combined_G, pos,
                              edgelist=neg_edges,
                              edge_color=neg_normalized,
                              edge_cmap=plt.cm.colors.ListedColormap(autumn_truncated),
                              width=3,
                              alpha=0.8,
                              arrows=True,
                              arrowsize=25,
                              arrowstyle='->')
    
    if pos_edges:
        nx.draw_networkx_edges(combined_G, pos,
                              edgelist=pos_edges,
                              edge_color=pos_normalized,
                              edge_cmap=plt.cm.colors.ListedColormap(summer_truncated),
                              width=3,
                              alpha=0.8,
                              arrows=True,
                              arrowsize=25,
                              arrowstyle='->')
    
    nx.draw_networkx_labels(combined_G, pos, 
                           font_size=15, 
                           font_weight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    if neg_weights and pos_weights:
        fig = plt.gcf()
        
        if neg_weights:
            sm_neg = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(autumn_truncated), 
                                         norm=plt.Normalize(vmin=min(neg_weights), 
                                                           vmax=max(neg_weights)))
            sm_neg.set_array([])
            cbar_neg = fig.colorbar(sm_neg, ax=plt.gca(), shrink=0.8, pad=0.01, location='right')
            cbar_neg.set_label(f'Negative ({metric_neg.replace("_", " ").title()})', 
                              rotation=270, labelpad=15, fontsize=12)
        
        if pos_weights:
            sm_pos = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(summer_truncated), 
                                         norm=plt.Normalize(vmin=min(pos_weights), 
                                                           vmax=max(pos_weights)))
            sm_pos.set_array([])
            cbar_pos = fig.colorbar(sm_pos, ax=plt.gca(), shrink=0.8, pad=0.03, location='right')
            cbar_pos.set_label(f'Positive ({metric_pos.replace("_", " ").title()})', 
                              rotation=270, labelpad=15, fontsize=12)
    
    elif neg_weights:
        sm_neg = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(autumn_truncated), 
                                     norm=plt.Normalize(vmin=min(neg_weights), 
                                                       vmax=max(neg_weights)))
        sm_neg.set_array([])
        cbar_neg = plt.colorbar(sm_neg, shrink=0.8, ax=plt.gca())
        cbar_neg.set_label(f'Negative ({metric_neg.replace("_", " ").title()})', 
                          rotation=270, labelpad=20)
    
    elif pos_weights:
        sm_pos = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(summer_truncated), 
                                     norm=plt.Normalize(vmin=min(pos_weights), 
                                                       vmax=max(pos_weights)))
        sm_pos.set_array([])
        cbar_pos = plt.colorbar(sm_pos, shrink=0.8, ax=plt.gca())
        cbar_pos.set_label(f'Positive ({metric_pos.replace("_", " ").title()})', 
                          rotation=270, labelpad=20)
    
    plt.suptitle(f"Combined Graph: Pos metric: {metric_pos.replace('_', ' ').title()}; Neg metric: {metric_neg.replace('_', ' ').title()}", 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs(SENTIMENT_GRAPH_DIR, exist_ok=True)
    output_file = os.path.join(SENTIMENT_GRAPH_DIR, f'combined_{metric_pos}_{metric_neg}.png')
    plt.savefig(output_file, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    metric_folder = os.path.join(SENTIMENT_GRAPH_DIR, f"combined_{metric_pos}_{metric_neg}")
    os.makedirs(metric_folder, exist_ok=True)
    
    edges_csv_path = os.path.join(metric_folder, 'edges.csv')
    nodes_csv_path = os.path.join(metric_folder, 'nodes.csv')
    
    export_graph_to_csv(combined_G, edges_csv_path)
    export_node_degrees_to_csv(combined_G, nodes_csv_path)


def create_and_visualize_graph(G, metric, sentiment='neg', top_n=100, show=False):
    """
    Create and visualize a graph for a specific metric and sentiment.
    
    Args:
        G: NetworkX graph to analyze
        metric: Metric to use for graph creation
        sentiment: Sentiment type to analyze ('neg' for negative sentiment -1, 'neu_pos' for neutral/positive sentiment 1)
        top_n: Number of top edges to keep for visualization
        show: If True, display the plot. If False, only save to file.
    """
    sentiment_G = create_sentiment_graph(G, metric=metric, sentiment=sentiment)
    
    if sentiment_G.number_of_edges() == 0:
        return
    
    filtered_G = filter_top_edges(sentiment_G, n_edges=top_n)
    
    if filtered_G.number_of_edges() == 0:
        return
    
    plt.figure(figsize=(20, 16))
    
    pos = nx.shell_layout(filtered_G)

    edges = filtered_G.edges()
    weights = [filtered_G[u][v]['weight'] for u, v in edges]
    
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5 for w in weights]
    else:
        normalized_weights = [0.5] * len(edges)
    
    if sentiment == 'neg':
        node_color = 'lightcoral'
        edge_cmap = plt.cm.autumn
    else:
        node_color = 'lightgreen'
        edge_cmap = plt.cm.summer
    
    node_sizes = [filtered_G.degree(node) * 400 + 600 for node in filtered_G.nodes()]
    nx.draw_networkx_nodes(filtered_G, pos, 
                          node_size=node_sizes,
                          node_color=node_color,
                          alpha=0.8)
    
    edges_drawn = nx.draw_networkx_edges(filtered_G, pos,
                                        edge_color=normalized_weights,
                                        edge_cmap=edge_cmap,
                                        width=3,
                                        alpha=0.8,
                                        arrows=True,
                                        arrowsize=25,
                                        arrowstyle='->')
    
    nx.draw_networkx_labels(filtered_G, pos, 
                           font_size=15, 
                           font_weight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    edge_labels = {}
    for u, v in filtered_G.edges():
        weight = filtered_G[u][v]['weight']
        edge_labels[(u, v)] = f"{weight:.2f}"
    
    if edges_drawn and weights:
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, 
                                 norm=plt.Normalize(vmin=min(weights), 
                                                   vmax=max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.8, ax=plt.gca())
        cbar.set_label(f'{metric.replace("_", " ").title()} Score', rotation=270, labelpad=20)
    
    metric_descriptions = METRIC_DESCRIPTIONS
    
    sentiment_label = "Negative" if sentiment == 'neg' else "Neutral/Positive"
    plt.title(f"Top {top_n} {sentiment_label} ({metric_descriptions[metric]})", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    os.makedirs(SENTIMENT_GRAPH_DIR, exist_ok=True)
    output_file = os.path.join(SENTIMENT_GRAPH_DIR, f'strong_{sentiment}_sentiment_{metric}.png')
    plt.savefig(output_file, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    metric_folder = os.path.join(SENTIMENT_GRAPH_DIR, f"{sentiment}_{metric}")
    os.makedirs(metric_folder, exist_ok=True)
    
    edges_csv_path = os.path.join(metric_folder, 'edges.csv')
    nodes_csv_path = os.path.join(metric_folder, 'nodes.csv')
    
    export_graph_to_csv(filtered_G, edges_csv_path)
    export_node_degrees_to_csv(filtered_G, nodes_csv_path)

def export_graph_to_csv(G, filename):
    """
    Export graph data to CSV format with columns: subreddit1, subreddit2, weight
    """
    csv_data = []
    for u, v, data in G.edges(data=True):
        csv_data.append({
            'subreddit1': u,
            'subreddit2': v,
            'weight': data['weight']
        })
    
    csv_data.sort(key=lambda x: x['weight'], reverse=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['subreddit1', 'subreddit2', 'weight']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

def export_node_degrees_to_csv(G, filename):
    """
    Export node degree information to CSV format with columns: subreddit, indegree, outdegree
    """
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    csv_data = []
    for node in G.nodes():
        csv_data.append({
            'subreddit': node,
            'indegree': in_degrees[node],
            'outdegree': out_degrees[node]
        })
    
    csv_data.sort(key=lambda x: (x['outdegree'], x['indegree']), reverse=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['subreddit', 'indegree', 'outdegree']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)


if __name__ == "__main__":
    os.makedirs(SENTIMENT_GRAPH_DIR, exist_ok=True)
    
    G = load_reddit_graph()
    
    # SENTIMENT METRICS
    sentiment_metrics = [
        'sentiment_frequency', 'relative_sentiment', 'sentiment_intensity', 
        'sentiment_count', 'source_total_sentiment', 'source_activity_weighted',
        'target_activity_weighted', 'combined_activity', 'subreddit_size_ratio', 
        'sentiment_ratio'
    ]
    
    # HATE-FOCUSED METRICS
    hate_metrics = [
        'direct_hate_score', 'aggressive_communication', 'emotional_toxicity',
        'social_conflict', 'comprehensive_hate', 'hate_intensity_weighted'
    ]
    
    all_metrics = sentiment_metrics + hate_metrics
    
    for metric in all_metrics:
        create_and_visualize_graph(G, metric, sentiment='neg', show=False)
    
    for metric in all_metrics:
        create_and_visualize_graph(G, metric, sentiment='neu_pos', show=False)
    
    for metric in hate_metrics:
        create_and_visualize_graph(G, metric, sentiment='neg', show=False)
    
    print("Creating combined sentiment graphs...")
    
    combined_metrics = [
        ('sentiment_ratio', 'sentiment_ratio'),
        ('sentiment_ratio', 'comprehensive_hate'),
        ('sentiment_ratio', 'emotional_toxicity'),
        ('sentiment_ratio', 'aggressive_communication'),
        ('sentiment_ratio', 'social_conflict')
    ]
    
    for metric_pos, metric_neg in combined_metrics:
        print(f"Creating combined graph: {metric_pos} (pos) vs {metric_neg} (neg)...")
        create_and_visualize_combined_graph(G, metric_pos=metric_pos, metric_neg=metric_neg, 
                                          top_n_neg=50, top_n_pos=50, show=False)
