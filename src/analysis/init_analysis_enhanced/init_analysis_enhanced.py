import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'enhanced_analysis_charts')

def load_dataset(file_path, strip_ids=True):
    """
    Load a dataset and optionally strip POST_ID to 6 characters.
    
    Args:
        file_path: Path to the dataset file
        strip_ids: Whether to strip POST_ID to 6 characters
    
    Returns:
        DataFrame with optionally stripped POST_ID
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, dtype={162: 'object'}) # Cast column 162 (mod_reason_by) to object
    else:
        df = load_reddit_df(file_path)
    
    if strip_ids and 'POST_ID' in df.columns:
        df['POST_ID'] = df['POST_ID'].astype(str).str[:6]
    
    return df

def get_id_sets(df1, df2):
    """
    Get ID sets and their intersection from two datasets.
    
    Args:
        df1, df2: DataFrames with POST_ID columns
    
    Returns:
        Tuple of (matching_ids, ids1, ids2)
    """
    ids1 = set(df1['POST_ID'].astype(str))
    ids2 = set(df2['POST_ID'].astype(str))
    matching_ids = ids1.intersection(ids2)
    return matching_ids, ids1, ids2

def create_comparison_plot(ids1, ids2, matching_ids, title, save_path=None, show=False):
    """
    Create a comparison plot with pie chart and bar chart.
    
    Args:
        ids1, ids2: Sets of IDs from two datasets
        matching_ids: Set of matching IDs
        title: Plot title
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, f'{title.lower().replace(" ", "_")}.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    labels = ['Overlap', 'Original Dataset Only']
    sizes = [len(matching_ids), len(ids2 - ids1)]
    colors = ['lightgreen', 'lightblue']
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('ID Overlap Distribution')
    
    categories = ['Original Dataset', 'Enhanced Dataset', 'Overlap']
    values = [len(ids1), len(ids2), len(matching_ids)]
    colors_bar = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = axes[1].bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[1].set_title('Dataset Size Comparison')
    axes[1].set_ylabel('Number of Unique IDs')
    
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def create_log_histogram(ax, data, title, xlabel, color):
    """
    Create a log-log histogram on the given axis.
    
    Args:
        ax: Matplotlib axis
        data: Data to plot
        title: Plot title
        xlabel: X-axis label
        color: Bar color
    """
    positive_data = data[data > 0]
    
    if len(positive_data) > 0:
        min_val = positive_data.min()
        max_val = positive_data.max()
        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
        
        ax.hist(positive_data, bins=log_bins, alpha=0.7, color=color, edgecolor='black')
        ax.set_title(f'{title} (Log-Log Scale)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No positive {title.lower()} found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

def create_enhanced_dataset_overview_plot(enhanced_df, save_path=None, show=False):
    """
    Create overview plots for the enhanced dataset showing voting distributions.
    
    Args:
        enhanced_df: Enhanced dataset DataFrame
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'enhanced_dataset_overview.png')
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle('Enhanced Dataset Overview', fontsize=16, fontweight='bold')
    
    upvotes = enhanced_df['ups']
    upvote_ratio = enhanced_df['upvote_ratio']
    
    total_votes = upvotes / upvote_ratio
    total_votes = total_votes.replace([np.inf, -np.inf], 0)
    downvotes = total_votes - upvotes
    num_comments = enhanced_df['num_comments']
    
    create_log_histogram(axes[0], upvotes, 'Upvotes Distribution', 'Upvotes', 'lightgreen')
    create_log_histogram(axes[1], downvotes, 'Downvotes Distribution', 'Downvotes', 'lightcoral')
    create_log_histogram(axes[2], total_votes, 'Total Votes Distribution', 'Total Votes', 'lightblue')
    create_log_histogram(axes[3], num_comments, 'Comments Distribution', 'Number of Comments', 'lightyellow')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def create_all_enhanced_analysis_plots(show=False):
    """
    Create all enhanced analysis plots.
    
    Args:
        show: Whether to display plots
    
    Returns:
        Tuple of datasets and matching IDs
    """
    enhanced_df = load_dataset(os.path.join(DATA_FOLDER, 'enhanced_dataset.csv'))
    reddit_df = load_dataset(os.path.join(DATA_FOLDER, 'reddit_df_merged.pkl'))
    
    matching_ids, enhanced_ids, reddit_ids = get_id_sets(enhanced_df, reddit_df)
    
    create_comparison_plot(enhanced_ids, reddit_ids, matching_ids, 
                          'Dataset ID Comparison', show=show)
    create_enhanced_dataset_overview_plot(enhanced_df, show=show)
    
    return enhanced_df, reddit_df, matching_ids

if __name__ == "__main__":
    create_all_enhanced_analysis_plots(show=True)
