from src.utils.data_utils import load_final_df
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

def plot_supercluster_sentiment(save=True, show=True):
    """
    Plot pie charts showing VADER compound sentiment distribution for each source supercluster.
    All charts are displayed in a single figure with subplots.
    
    Args:
        save: If True, saves sentiment distribution data as CSV file in figures directory
        show: If True, displays the plot
    """
    df = load_final_df()
    
    df['Sentiment_compound_vader'] = pd.to_numeric(df['Sentiment_compound_vader'], errors='coerce')
    
    df = df[df['Sentiment_compound_vader'].notna() & df['source_supercluster'].notna()]
    df['sentiment_category'] = df['Sentiment_compound_vader'].apply(
        lambda x: 'Negative' if x < -0.05 else ('Neutral' if x <= 0.05 else 'Positive')
    )
    
    superclusters = sorted(df['source_supercluster'].dropna().unique())
    n_clusters = len(superclusters)
    
    csv_data = {'': ['Positive', 'Neutral', 'Negative']}
    
    cols = 3
    rows = math.ceil(n_clusters / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
    
    colors = {'Negative': '#ff6b6b', 'Neutral': '#ffd93d', 'Positive': '#6bcf7f'}
    
    for idx, supercluster in enumerate(superclusters):
        ax = axes[idx]
        cluster_data = df[df['source_supercluster'] == supercluster]
        sentiment_counts = cluster_data['sentiment_category'].value_counts()
        
        positive_count = sentiment_counts.get('Positive', 0)
        neutral_count = sentiment_counts.get('Neutral', 0)
        negative_count = sentiment_counts.get('Negative', 0)
        total = len(cluster_data)
        
        csv_data[supercluster] = [positive_count, neutral_count, negative_count]
        
        color_list = [colors.get(cat, '#95a5a6') for cat in sentiment_counts.index]
        
        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=color_list,
            textprops={'fontsize': 10}
        )
        
        ax.set_title(f'{supercluster}\n(n={total:,})', fontsize=11, pad=10)
    
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save:
        csv_df = pd.DataFrame(csv_data)
        figures_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'cluster_sentiment')
        os.makedirs(figures_dir, exist_ok=True)
        csv_path = os.path.join(figures_dir, 'supercluster_sentiment_distribution.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved sentiment data to: {csv_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    plot_supercluster_sentiment(save=True, show=True)

