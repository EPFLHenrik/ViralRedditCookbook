import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.data_utils import load_final_df


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "virality_analysis")


def get_viral_posts(df, virality_percentile=0.95, window_size='60D'):
    """
    Calculate moving average and moving standard deviation for virality_rss with a window_size window per cluster,
    then mark posts as viral if their z-score relative to the moving average exceeds the percentile threshold.
    
    Args:
        df: DataFrame produced by load_final_df with virality_rss score calculated
        virality_percentile: Percentile threshold for determining viral posts (default: 0.95 = top 5%)
        window_size: Window size for moving average and standard deviation (default: '60D' = 60 days)
    
    Returns:
        pandas.DataFrame: DataFrame with added columns:
            - moving_avg_virality: window_size moving average of virality_rss (per cluster)
            - moving_std_virality: window_size moving standard deviation of virality_rss (per cluster)
            - z_score: z-score relative to moving average: (value - moving_avg) / moving_std
            - is_viral: 1 if post's z-score is in top p% of posts above moving average (z > 0) for its cluster, 0 otherwise
    """
    
    df = df.copy()
    
    df['moving_avg_virality'] = np.nan
    df['moving_std_virality'] = np.nan
    df['z_score'] = np.nan
    df['is_viral'] = False
    
    indices_to_drop = []
    
    for cluster in df['source_supercluster'].unique():
        cluster_mask = df['source_supercluster'] == cluster
        cluster_df = df[cluster_mask].copy()
        
        cluster_df['original_index'] = cluster_df.index
        
        cluster_df = cluster_df.sort_values('TIMESTAMP').reset_index(drop=True)
        
        cluster_indexed = cluster_df.set_index('TIMESTAMP')
        
        cluster_indexed['moving_avg_virality'] = cluster_indexed['virality_rss'].rolling(
            window=window_size, min_periods=1
        ).mean()
        cluster_indexed['moving_std_virality'] = cluster_indexed['virality_rss'].rolling(
            window=window_size, min_periods=2
        ).std()
        
        cluster_df = cluster_indexed.reset_index()
        
        first_datapoint_index = cluster_df.iloc[0]['original_index']
        indices_to_drop.append(first_datapoint_index) # remove the first datapoint because it has NaN std
        cluster_df = cluster_df.iloc[1:].reset_index(drop=True)
        
        cluster_original_indices = cluster_df['original_index'].values
        
        df.loc[cluster_original_indices, 'moving_avg_virality'] = cluster_df['moving_avg_virality'].values
        df.loc[cluster_original_indices, 'moving_std_virality'] = cluster_df['moving_std_virality'].values
        
        moving_avg = cluster_df['moving_avg_virality'].values
        moving_std = cluster_df['moving_std_virality'].values
        z_scores = (cluster_df['virality_rss'].values - moving_avg) / moving_std
        
        df.loc[cluster_original_indices, 'z_score'] = z_scores
        
        n_posts = len(z_scores)
        n_viral = int(np.ceil(n_posts * (1 - virality_percentile)))
        
        top_indices = np.argsort(z_scores)[-n_viral:]
        
        viral_mask = np.zeros(len(z_scores), dtype=bool)
        viral_mask[top_indices] = True
        
        viral_indices = cluster_original_indices[viral_mask]
        df.loc[viral_indices, 'is_viral'] = True
    
    if len(indices_to_drop) > 0:
        df = df.drop(index=indices_to_drop)
    
    return df


def plot_viral_posts(df, virality_percentile=0.95, window_size='60D', show=False):
    """
    Plot moving average of virality_rss as curves and viral posts as points above the curves,
    with separate subplots for each cluster in a single figure.
    
    Args:
        df: DataFrame with virality_rss calculated. If None, loads using load_final_df
        virality_percentile: Percentile threshold for viral posts (default: 0.95)
        window_size: Window size for moving average and standard deviation (default: '60D' = 60 days)
        show: Whether to display the plot (default: False)
    """
        
    clusters = sorted(df['source_supercluster'].dropna().unique())
    n_clusters = len(clusters)
    
    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig.suptitle(f'Viral Posts Over Time by Cluster (Moving Average with {virality_percentile*100:.1f}% Threshold) {window_size} window', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        cluster_df = df[df['source_supercluster'] == cluster].copy()
        
        cluster_df = cluster_df.sort_values('TIMESTAMP').reset_index(drop=True)
        
        ax.plot(cluster_df['TIMESTAMP'], cluster_df['moving_avg_virality'], color='blue', linewidth=2, label='Moving Average (window_size window)', alpha=0.7)
        
        viral_posts = cluster_df[cluster_df['is_viral'] == True].copy()
        
        ax.scatter(viral_posts['TIMESTAMP'], viral_posts['virality_rss'],
                    color='red', s=30, alpha=0.6, 
                    label=f'Viral Posts (top {virality_percentile*100:.1f}%)', 
                    marker='o', zorder=5)
        
        upper_bound = cluster_df['moving_avg_virality'] + cluster_df['moving_std_virality'].fillna(0)
        lower_bound = cluster_df['moving_avg_virality'] - cluster_df['moving_std_virality'].fillna(0)
        ax.fill_between(cluster_df['TIMESTAMP'], lower_bound, upper_bound, alpha=0.2, color='blue', label='Moving Average +/- 1 Std Dev')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Virality RSS')
        ax.set_yscale('log')
        ax.set_title(f'{cluster}')
        ax.grid(True, alpha=0.3)
        
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        save_path = os.path.join(FIGURES_DIR, f'viral_posts_all_clusters_{virality_percentile*100:.1f}percent_{window_size}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    df = load_final_df(calculate_rss=True)
    df = get_viral_posts(df, virality_percentile=0.95, window_size='60D')
    plot_viral_posts(df, virality_percentile=0.95, window_size='60D', show=False)
