import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from src.utils.data_utils import load_reddit_df, load_enhanced_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
CHARTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'init_analysis_enhanced_charts')


def get_id_sets(reddit_df, enhanced_df):
    """
    Get ID sets and their intersection from two datasets.
    
    Args:
        reddit_df, enhanced_df: DataFrames with POST_ID columns
    
    Returns:
        Tuple of (matching_ids, reddit_ids, enhanced_ids)
    """
    reddit_ids = set(reddit_df['POST_ID'].astype(str))
    enhanced_ids = set(enhanced_df['POST_ID'].astype(str))
    matching_ids = reddit_ids.intersection(enhanced_ids)
    return matching_ids, reddit_ids, enhanced_ids

def create_comparison_plot(reddit_ids, cleaned_ids, matching_ids, title, save_path=None, show=False):
    """
    Create a comparison plot with pie chart and bar chart.
    
    Args:
        reddit_ids: Set of IDs from original dataset
        cleaned_ids: Set of IDs from cleaned dataset
        matching_ids: Set of matching IDs between original and cleaned datasets
        title: Plot title
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, f'{title.lower().replace(" ", "_")}.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    labels = ['In Cleaned Dataset', 'Not in Cleaned Dataset']
    sizes = [len(matching_ids), len(reddit_ids - cleaned_ids)]
    colors = ['lightgreen', 'lightcoral']
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Original Dataset Coverage')
    
    categories = ['Original Dataset', 'Scrapped data', 'Overlap']
    values = [len(reddit_ids), len(cleaned_ids), len(matching_ids)]
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

def create_enhanced_dataset_overview_plot(cleaned_df, save_path=None, show=False):
    """
    Create overview plots for the cleaned dataset showing voting distributions.
    
    Args:
        cleaned_df: Cleaned dataset DataFrame
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if save_path is None:
        save_path = os.path.join(CHARTS_DIR, 'enhanced_dataset_overview.png')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cleaned Dataset Overview', fontsize=16, fontweight='bold')
    
    upvotes = cleaned_df['ups']
    upvote_ratio = cleaned_df['upvote_ratio']
    
    total_votes = upvotes / upvote_ratio
    total_votes = total_votes.replace([np.inf, -np.inf], 0)
    downvotes = total_votes - upvotes
    num_comments = cleaned_df['num_comments']
    
    create_log_histogram(axes[0, 0], upvotes, 'Upvotes Distribution', 'Upvotes', 'lightgreen')
    create_log_histogram(axes[0, 1], downvotes, 'Downvotes Distribution', 'Downvotes', 'lightcoral')
    create_log_histogram(axes[1, 0], total_votes, 'Total Votes Distribution', 'Total Votes', 'lightblue')
    create_log_histogram(axes[1, 1], num_comments, 'Comments Distribution', 'Number of Comments', 'lightyellow')
    
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
    enhanced_df = load_enhanced_df()
    reddit_df = load_reddit_df()
    
    enhanced_df_cleaned = enhanced_df.dropna(subset=['ups', 'downs', 'num_comments'])
    
    matching_ids, reddit_ids, enhanced_ids = get_id_sets(reddit_df, enhanced_df)
    
    cleaned_ids = set(enhanced_df_cleaned['POST_ID'].astype(str))
    reddit_cleaned_matching_ids = reddit_ids.intersection(cleaned_ids)
    
    create_comparison_plot(reddit_ids, cleaned_ids, reddit_cleaned_matching_ids, 'Scraped data vs original data comparison', show=show)
    
    create_enhanced_dataset_overview_plot(enhanced_df_cleaned, show=show)
    
    return enhanced_df, reddit_df, enhanced_df_cleaned, reddit_ids, enhanced_ids, matching_ids, reddit_cleaned_matching_ids


if __name__ == "__main__":
    create_all_enhanced_analysis_plots(show=True)
