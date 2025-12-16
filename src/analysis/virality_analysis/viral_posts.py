import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures", "virality_analysis", "viral_posts")


def save_coefficients_to_csv(coef_subset, filename='viral_posts.csv'):
    """
    Save logistic regression coefficients to CSV with only specified columns.
    Columns saved: Feature, Coefficient, P-value, Abs_Coefficient
    Feature is the first column (like "Name" in the spreadsheet format).
    
    Args:
        coef_subset: DataFrame with columns Feature, Coefficient, P-value, Abs_Coefficient
        filename: Output filename (default: 'viral_posts.csv')
    """
    
    coef_subset = coef_subset.copy()
    coef_subset['virality_rss'] = coef_subset['virality_rss'].apply(lambda x: f"{x:.2f}")

    coef_subset.rename(columns={'Automated_readability_index': 'AR_index'}, inplace=True)
    coef_subset.rename(columns={'Negated_Sentiment_compound_vader': 'Negated_sentiment_comp'}, inplace=True)
    coef_subset.rename(columns={'Negated_Avg_word_length': 'Negated_word_length'}, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, filename)
    coef_subset.to_csv(output_path, index=False)


def plot_spider_charts(coef_subset, n_cols=3):
    """
    Create spider/radar charts for each row in the dataframe.
    Intended to be called after preprocessing with percentile scaling and negation.
    Automatically detects positive and negated columns from the dataframe.
    
    Args:
        coef_subset: DataFrame with preprocessed data (already scaled and negated). 
                    Must have 'title' column. Excludes 'title' and 'virality_rss' from plotting.
        n_cols: Number of columns in the grid (default: 3)
    """
    exclude_cols = {'title', 'virality_rss'}
    all_cols = [col for col in coef_subset.columns if col not in exclude_cols]
    
    positive_columns = [col for col in all_cols if not col.startswith('Negated_')]
    negated_columns = [col for col in all_cols if col.startswith('Negated_')]
    
    columns_to_plot = positive_columns + negated_columns
    
    if len(columns_to_plot) == 0:
        raise ValueError("No valid columns found to plot")
    
    n_plots = len(coef_subset)
    
    n_grid_rows = math.ceil(n_plots / n_cols)
    
    figsize = (5 * n_cols, 4 * n_grid_rows)
    
    fig = plt.figure(figsize=figsize)
    
    num_vars = len(columns_to_plot)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, (_, row) in enumerate(coef_subset.iterrows()):
        ax = fig.add_subplot(n_grid_rows, n_cols, idx + 1, projection='polar')
        
        values = [row[col] for col in columns_to_plot]
        values += values[:1]
        
        ax.plot(angles, values, 'o-')
        ax.fill(angles, values)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(columns_to_plot)
        
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        
        title = row.get('title', f'Post {idx + 1}')
        if len(title) > 60:
            title = title[:57] + '...'
        ax.set_title(title)
    
    for idx in range(n_plots, n_grid_rows * n_cols):
        ax = fig.add_subplot(n_grid_rows, n_cols, idx + 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

