import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import os

from src.utils.data_utils import load_reddit_graph

CHARTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'init_analysis_charts')



def moving_average(data, window_size):
    """
    Calculate moving average for smoothing time series data
    
    Args:
        data: Time series data to be smoothed
        window_size: Size of the moving average window
    
    Returns:
        Smoothed time series data with moving average applied
    """
    if window_size <= 1 or len(data) < window_size:
        return data
    
    data = np.array(data, dtype=float)
    smoothed = np.zeros_like(data)
    
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed[i] = np.mean(data[start_idx:end_idx])
    
    return smoothed


def compute_ccdf(degrees):
        """
        Helper function to compute CCDF (Complementary Cumulative Distribution Function)
        
        Args:
            degrees: Array of degree values to compute CCDF for
        
        Returns:
            sorted_degrees: Sorted degree values
            ccdf: Complementary cumulative distribution function values
        """
        
        sorted_degrees = np.sort(degrees)
        n = len(sorted_degrees)
        ccdf = 1 - np.cumsum(np.ones(n)) / n
        return sorted_degrees, ccdf


def plot_degree_distribution(G, save_path=None, show=False):
    """
    Plot in-degree and out-degree distributions including log-log scale and CCDF
    
    Args:
        G: Directed graph to analyze degree distributions for
        save_path: Path to save the plot. If None, saves to default location CHARTS_DIR
        show: If True, displays the plot. If False, saves it to file
    """

    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'degree_distribution.png')
    
    in_degrees = np.array([d for n, d in G.in_degree()])
    out_degrees = np.array([d for n, d in G.out_degree()])
    
    in_degrees_sorted, in_ccdf = compute_ccdf(in_degrees)
    out_degrees_sorted, out_ccdf = compute_ccdf(out_degrees)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Degree Distribution Analysis', fontsize=16, fontweight='bold')
    
    in_degrees_nonzero = in_degrees[in_degrees > 0]
    out_degrees_nonzero = out_degrees[out_degrees > 0]
    
    # Plot indegree histogram with bins of equal width in log space
    log_bins_in = np.logspace(np.log10(in_degrees_nonzero.min()), 
                                np.log10(in_degrees_nonzero.max()), 50)
    axes[0].hist(in_degrees_nonzero, bins=log_bins_in, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('In-Degree Distribution (Log Log Scale)')
    axes[0].set_xlabel('In-Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Plot outdegree histogram with bins of equal width in log space
    log_bins_out = np.logspace(np.log10(out_degrees_nonzero.min()), 
                                np.log10(out_degrees_nonzero.max()), 50)
    axes[1].hist(out_degrees_nonzero, bins=log_bins_out, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title('Out-Degree Distribution (Log Log Scale)')
    axes[1].set_xlabel('Out-Degree')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Plot CCDF comparrison
    axes[2].plot(in_degrees_sorted, in_ccdf, 'b-', linewidth=2, alpha=0.8, label='In-Degree')
    axes[2].plot(out_degrees_sorted, out_ccdf, 'r-', linewidth=2, alpha=0.8, label='Out-Degree')
    axes[2].set_title('CCDF Comparison')
    axes[2].set_xlabel('Degree')
    axes[2].set_ylabel('CCDF')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def plot_sentiment_analysis(G, save_path=None, show=False):
    """
    Analyze and visualize sentiment patterns
    
    Args:
        G: Directed graph with edges containing sentiment data
        save_path: Path to save the plot. If None, saves to default location CHARTS_DIR
        show: If True, displays the plot. If False, saves it to file
    """

    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'sentiment_analysis.png')
    
    sentiments = []
    for u, v, data in G.edges(data=True):
        sentiments.append(data['sentiment'])
    
    sentiment_counts = Counter(sentiments)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Sentiment Analysis', fontsize=16, fontweight='bold')
    
    labels = ['Neutral/Positive (1)', 'Negative (-1)']
    sizes = [sentiment_counts[1], sentiment_counts[-1]]
    colors = ['lightgreen', 'lightcoral']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, startangle=90)
    axes[0, 0].set_title('Overall Sentiment Distribution')
    
    axes[0, 1].bar(labels, sizes, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Sentiment Counts')
    axes[0, 1].set_ylabel('Number of Links')
    axes[0, 1].grid(True, alpha=0.3)
    
    pos_sentiment_by_source = {}
    neg_sentiment_by_source = {}
    
    for u, v, data in G.edges(data=True):
        if data['sentiment'] == 1:
            pos_sentiment_by_source[u] = pos_sentiment_by_source.get(u, 0) + 1
        elif data['sentiment'] == -1:
            neg_sentiment_by_source[u] = neg_sentiment_by_source.get(u, 0) + 1
    
    top_pos = sorted(pos_sentiment_by_source.items(), key=lambda x: x[1], reverse=True)[:10]
    subreddits_pos, counts_pos = zip(*top_pos)
    axes[1, 0].barh(range(len(subreddits_pos)), counts_pos, color='lightgreen', alpha=0.7)
    axes[1, 0].set_yticks(range(len(subreddits_pos)))
    axes[1, 0].set_yticklabels(subreddits_pos, fontsize=8)
    axes[1, 0].set_title('Top 10 Subreddits by Neutral/Positive Links')
    axes[1, 0].set_xlabel('Number of Neutral/Positive Links')
    axes[1, 0].grid(True, alpha=0.3)
    
    top_neg = sorted(neg_sentiment_by_source.items(), key=lambda x: x[1], reverse=True)[:10]
    subreddits_neg, counts_neg = zip(*top_neg)
    axes[1, 1].barh(range(len(subreddits_neg)), counts_neg, color='lightcoral', alpha=0.7)
    axes[1, 1].set_yticks(range(len(subreddits_neg)))
    axes[1, 1].set_yticklabels(subreddits_neg, fontsize=8)
    axes[1, 1].set_title('Top 10 Subreddits by Negative Links')
    axes[1, 1].set_xlabel('Number of Negative Links')
    axes[1, 1].grid(True, alpha=0.3)
    
    pos_percentage_by_source = {}
    for subreddit in set(list(pos_sentiment_by_source.keys()) + list(neg_sentiment_by_source.keys())):
        pos_count = pos_sentiment_by_source.get(subreddit, 0)
        neg_count = neg_sentiment_by_source.get(subreddit, 0)
        total_count = pos_count + neg_count
        if total_count > 1000:
            pos_percentage_by_source[subreddit] = (pos_count / total_count) * 100
    
    top_pos_pct = sorted(pos_percentage_by_source.items(), key=lambda x: x[1], reverse=True)[:10]
    subreddits_pos_pct, percentages_pos = zip(*top_pos_pct)
    axes[2, 0].barh(range(len(subreddits_pos_pct)), percentages_pos, color='lightgreen', alpha=0.7)
    axes[2, 0].set_yticks(range(len(subreddits_pos_pct)))
    axes[2, 0].set_yticklabels(subreddits_pos_pct, fontsize=8)
    axes[2, 0].set_title('Top 10 Subreddits by Percentage of Neutral/Positive Links (exclude subreddits w/ <1000 posts)')
    axes[2, 0].set_xlabel('Percentage of Links with Neutral/Positive Sentiment (%)')
    axes[2, 0].grid(True, alpha=0.3)
    
    neg_percentage_by_source = {}
    for subreddit in set(list(pos_sentiment_by_source.keys()) + list(neg_sentiment_by_source.keys())):
        pos_count = pos_sentiment_by_source.get(subreddit, 0)
        neg_count = neg_sentiment_by_source.get(subreddit, 0)
        total_count = pos_count + neg_count
        if total_count > 1000:
            neg_percentage_by_source[subreddit] = (neg_count / total_count) * 100
    
    top_neg_pct = sorted(neg_percentage_by_source.items(), key=lambda x: x[1], reverse=True)[:10]
    subreddits_neg_pct, percentages_neg = zip(*top_neg_pct)
    axes[2, 1].barh(range(len(subreddits_neg_pct)), percentages_neg, color='lightcoral', alpha=0.7)
    axes[2, 1].set_yticks(range(len(subreddits_neg_pct)))
    axes[2, 1].set_yticklabels(subreddits_neg_pct, fontsize=8)
    axes[2, 1].set_title('Top 10 Subreddits by Percentage of Negative Links (exclude subreddits w/ <1000 posts)')
    axes[2, 1].set_xlabel('Percentage of Links with Negative Sentiment (%)')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

def plot_temporal_analysis(G, save_path=None, show=False):
    """
    Analyze temporal patterns in the data
    
    Args:
        G: Directed graph with edges containing timestamp and sentiment data
        save_path: Path to save the plot. If None, saves to default location CHARTS_DIR
        show: If True, displays the plot. If False, saves it to file
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'temporal_analysis.png')
    
    timestamps = []
    sentiments_time = []
    
    for u, v, data in G.edges(data=True):
        timestamp_str = data['timestamp']
        if pd.notna(timestamp_str):
            timestamp = pd.to_datetime(timestamp_str)
            timestamps.append(timestamp)
            sentiments_time.append(data['sentiment'])
    
    df_time = pd.DataFrame({
        'timestamp': timestamps,
        'sentiment': sentiments_time
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Temporal Analysis', fontsize=16, fontweight='bold')
    
    df_time['date'] = df_time['timestamp'].dt.date
    posts_per_day = df_time.groupby('date').size()
    
    posts_per_day = posts_per_day.sort_index()
    
    window_size = 50  # 50-day window
    smoothed_values = moving_average(posts_per_day.values, window_size)
    
    dates = list(posts_per_day.index)
    axes[0, 0].plot(dates, posts_per_day.values, alpha=0.8, linewidth=1, color='lightblue', label='Original')
    axes[0, 0].plot(dates, smoothed_values, alpha=1, linewidth=2, color='darkblue', label='Smoothed')
    axes[0, 0].set_title('Posts Over Time (Smoothed)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Posts')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()
    
    df_time['hour'] = df_time['timestamp'].dt.hour
    posts_per_hour = df_time.groupby('hour').size()
    
    hours = list(posts_per_hour.index)
    axes[0, 1].bar(hours, posts_per_hour.values, alpha=0.7, color='orange')
    axes[0, 1].set_title('Posts by Hour of Day')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Number of Posts')
    axes[0, 1].set_xticks(range(0, 24, 2))
    
    df_time['day_of_week'] = df_time['timestamp'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    posts_per_dow = df_time.groupby('day_of_week').size().reindex(day_order)
    
    axes[1, 0].bar(range(len(posts_per_dow)), posts_per_dow.values, alpha=0.7, color='lightblue')
    axes[1, 0].set_title('Posts by Day of Week')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Number of Posts')
    axes[1, 0].set_xticks(range(len(day_order)))
    axes[1, 0].set_xticklabels(day_order, rotation=45)
    
    df_time['year_month'] = df_time['timestamp'].dt.to_period('M')
    sentiment_over_time = df_time.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    
    sentiment_over_time = sentiment_over_time.sort_index()
    
    window_size = 3  # 3-month window
    positive_smoothed = moving_average(sentiment_over_time[1].values, window_size)
    negative_smoothed = moving_average(sentiment_over_time[-1].values, window_size)
    
    month_labels = [str(x) for x in sentiment_over_time.index]
    positive_values = sentiment_over_time[1].values
    negative_values = sentiment_over_time[-1].values
    
    axes[1, 1].plot(month_labels, positive_values, 
                    alpha=0.8, linewidth=1, color='lightgreen', marker='o', markersize=3, label='Positive (Original)')
    axes[1, 1].plot(month_labels, positive_smoothed, 
                    alpha=0.8, linewidth=2, color='darkgreen', marker='o', label='Positive (Smoothed)')
    
    axes[1, 1].plot(month_labels, negative_values, 
                    alpha=0.8, linewidth=1, color='lightcoral', marker='s', markersize=3, label='Negative (Original)')
    axes[1, 1].plot(month_labels, negative_smoothed, 
                    alpha=0.8, linewidth=2, color='darkred', marker='s', label='Negative (Smoothed)')
    
    axes[1, 1].set_title('Sentiment Over Time (Monthly, Smoothed)')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Posts')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    tick_positions = range(0, len(month_labels), 3)  # Every 3rd month
    axes[1, 1].set_xticks(tick_positions)
    axes[1, 1].set_xticklabels([month_labels[i] for i in tick_positions], rotation=45)
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

def plot_property_vector_analysis(G, save_path=None, show=False):
    """
    Analyze and visualize the 86-element property vector from Reddit cross-linking posts
    
    Args:
        G: Directed graph with edges containing property vectors (86 features)
        save_path: Path to save the plot. If None, saves to default location CHARTS_DIR
        show: If True, displays the plot. If False, saves it to file
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'property_vector_analysis.png')
    
    property_names = [
        "Characters", "Characters (no spaces)", "Fraction alphabetical", "Fraction digits", 
        "Fraction uppercase", "Fraction whitespace", "Fraction special chars", "Word count",
        "Unique words", "Long words (6+ chars)", "Avg word length", "Stopwords count",
        "Fraction stopwords", "Sentence count", "Long sentences (10+ words)", 
        "Avg chars per sentence", "Avg words per sentence", "Readability index",
        "VADER positive", "VADER negative", "VADER compound", "LIWC_Funct", "LIWC_Pronoun",
        "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron",
        "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future",
        "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers",
        "LIWC_Swear", "LIWC_Social", "LIWC_Family", "LIWC_Friends", "LIWC_Humans", "LIWC_Affect",
        "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech",
        "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib",
        "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel",
        "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ",
        "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure",
        "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death", "LIWC_Assent", "LIWC_Dissent",
        "LIWC_Nonflu", "LIWC_Filler"
    ]
    
    property_vectors = []
    valid_edges = 0
    
    for u, v, data in G.edges(data=True):
        if 'properties' in data and data['properties']:
            props = data['properties']
            if isinstance(props, list) and len(props) == 86:
                property_vectors.append(props)
                valid_edges += 1
    
    property_array = np.array(property_vectors)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Property Vector Analysis (86 Features)', fontsize=16, fontweight='bold')
    
    vader_props = property_array[:, 18:21]
    vader_names = property_names[18:21]
    vader_means = np.mean(vader_props, axis=0)
    
    axes[0, 0].bar(vader_names, vader_means, alpha=0.7, color=['green', 'red', 'blue'])
    axes[0, 0].set_title('VADER Sentiment Analysis')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    liwc_emotional = property_array[:, 47:53]
    liwc_emotional_names = property_names[47:53]
    liwc_emotional_means = np.mean(liwc_emotional, axis=0)
    
    axes[0, 1].bar(liwc_emotional_names, liwc_emotional_means, alpha=0.7, color='purple')
    axes[0, 1].set_title('LIWC Emotional Features')
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    liwc_social = property_array[:, 43:46]
    liwc_social_names = property_names[43:46]
    liwc_social_means = np.mean(liwc_social, axis=0)
    
    axes[0, 2].bar(liwc_social_names, liwc_social_means, alpha=0.7, color='brown')
    axes[0, 2].set_title('LIWC Social Features (Log Scale)')
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_ylabel('Mean Value')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    liwc_perceptual = property_array[:, 62:66]
    liwc_perceptual_names = property_names[62:66]
    liwc_perceptual_means = np.mean(liwc_perceptual, axis=0)
    
    axes[1, 0].bar(liwc_perceptual_names, liwc_perceptual_means, alpha=0.7, color='pink')
    axes[1, 0].set_title('LIWC Perceptual Features')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    liwc_biological = property_array[:, 66:71]
    liwc_biological_names = property_names[66:71]
    liwc_biological_means = np.mean(liwc_biological, axis=0)
    
    axes[1, 1].bar(liwc_biological_names, liwc_biological_means, alpha=0.7, color='cyan')
    axes[1, 1].set_title('LIWC Biological Features')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    liwc_temporal = property_array[:, 71:75]
    liwc_temporal_names = property_names[71:75]
    liwc_temporal_means = np.mean(liwc_temporal, axis=0)
    
    axes[1, 2].bar(liwc_temporal_names, liwc_temporal_means, alpha=0.7, color='yellow')
    axes[1, 2].set_title('LIWC Temporal/Spatial Features')
    axes[1, 2].set_ylabel('Mean Value')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    liwc_work = property_array[:, 75:78]
    liwc_work_names = property_names[75:78]
    liwc_work_means = np.mean(liwc_work, axis=0)
    
    axes[2, 0].bar(liwc_work_names, liwc_work_means, alpha=0.7, color='gray')
    axes[2, 0].set_title('LIWC Work/Achievement Features')
    axes[2, 0].set_ylabel('Mean Value')
    axes[2, 0].tick_params(axis='x', rotation=45)
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def plot_basic_text_properties(G, save_path=None, show=False):
    """
    Plot basic text properties with feature names on x-axis
    
    Args:
        G: Directed graph with edges containing property vectors (86 features)
        save_path: Path to save the plot. If None, saves to default location CHARTS_DIR
        show: If True, displays the plot. If False, saves it to file
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'basic_text_properties.png')
    
    property_names = [
        "Characters", "Characters (no spaces)", "Fraction alphabetical", "Fraction digits", 
        "Fraction uppercase", "Fraction whitespace", "Fraction special chars", "Word count",
        "Unique words", "Long words (6+ chars)", "Avg word length", "Stopwords count",
        "Fraction stopwords", "Sentence count", "Long sentences (10+ words)", 
        "Avg chars per sentence", "Avg words per sentence", "Readability index"
    ]
    
    property_vectors = []
    
    for u, v, data in G.edges(data=True):
        if 'properties' in data and data['properties']:
            props = data['properties']
            if isinstance(props, list) and len(props) == 86:
                property_vectors.append(props[:18])
    
    property_array = np.array(property_vectors)
    basic_means = np.mean(property_array, axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Mean Values: Basic Text Properties (Log Scale)', fontsize=14, fontweight='bold')
    
    ax.bar(range(len(basic_means)), basic_means, alpha=0.7, color='skyblue')
    ax.set_xlabel('Property Name', fontsize=11)
    ax.set_ylabel('Mean Value (Log Scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(range(len(property_names)))
    ax.set_xticklabels(property_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def plot_linguistic_features(G, save_path=None, show=False):
    """
    Plot LIWC linguistic features with feature names on x-axis
    
    Args:
        G: Directed graph with edges containing property vectors (86 features)
        save_path: Path to save the plot. If None, saves to default location CHARTS_DIR
        show: If True, displays the plot. If False, saves it to file
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'linguistic_features.png')
    
    property_names = [
        "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", 
        "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", 
        "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", 
        "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers"
    ]
    
    property_vectors = []
    
    for u, v, data in G.edges(data=True):
        if 'properties' in data and data['properties']:
            props = data['properties']
            if isinstance(props, list) and len(props) == 86:
                property_vectors.append(props[21:42])
    
    property_array = np.array(property_vectors)
    linguistic_means = np.mean(property_array, axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle('Mean Values: LIWC Linguistic Features (Log Scale)', fontsize=14, fontweight='bold')
    
    ax.bar(range(len(linguistic_means)), linguistic_means, alpha=0.7, color='orange')
    ax.set_xlabel('Feature Name', fontsize=11)
    ax.set_ylabel('Mean Value (Log Scale)', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(range(len(property_names)))
    ax.set_xticklabels(property_names, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    G = load_reddit_graph()
    
    plot_degree_distribution(G)
    plot_sentiment_analysis(G)
    plot_temporal_analysis(G)
    plot_property_vector_analysis(G)
