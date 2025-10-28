import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_utils import load_enhanced_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')


def analyze_subreddit_overlap():
    enhanced_df = load_enhanced_df()
    
    embeddings_path = os.path.join(DATA_FOLDER, 'web-redditEmbeddings-subreddits.csv')
    embeddings_df = pd.read_csv(embeddings_path, header=None)
    embeddings_df.columns = ['subreddit'] + [f'dim_{i}' for i in range(len(embeddings_df.columns)-1)]
    
    enhanced_subreddits = set(enhanced_df['SOURCE_SUBREDDIT'].unique())
    embedding_subreddits = set(embeddings_df['subreddit'].unique())
    
    overlap = enhanced_subreddits & embedding_subreddits
    only_enhanced = enhanced_subreddits - embedding_subreddits
    only_embeddings = embedding_subreddits - enhanced_subreddits
    
    pct_enhanced_with_embeddings = len(overlap) / len(enhanced_subreddits) * 100
    pct_embeddings_in_enhanced = len(overlap) / len(embedding_subreddits) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    categories = ['Enhanced Posts\nDataset\n(Subreddits)', 'Only in Enhanced\nPosts Dataset', 'Overlap', 'Embeddings\nOnly']
    values = [len(enhanced_subreddits), len(only_enhanced), len(overlap), len(only_embeddings)]
    colors = ['#3498db', '#f39c12', '#2ecc71', '#e74c3c']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Subreddits', fontsize=12, fontweight='bold')
    ax1.set_title('Subreddit Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2 = axes[1]
    labels = [f'Subreddits with embeddings\n({pct_enhanced_with_embeddings:.1f}%)',
              f'Subreddits without embeddings\n({100-pct_enhanced_with_embeddings:.1f}%)']
    sizes = [len(overlap), len(only_enhanced)]
    colors_pie = ['#2ecc71', '#95a5a6']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                                         colors=colors_pie, explode=explode,
                                         startangle=90, textprops={'fontsize': 10})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax2.set_title('Coverage of Enhanced Posts Dataset\nby Embeddings', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def analyze_post_coverage():
    enhanced_df = load_enhanced_df()
    
    embeddings_path = os.path.join(DATA_FOLDER, 'web-redditEmbeddings-subreddits.csv')
    embeddings_df = pd.read_csv(embeddings_path, header=None)
    embeddings_df.columns = ['subreddit'] + [f'dim_{i}' for i in range(len(embeddings_df.columns)-1)]
    
    embedding_subreddits = set(embeddings_df['subreddit'].unique())
    
    total_posts = len(enhanced_df)
    filtered_df = enhanced_df[enhanced_df['SOURCE_SUBREDDIT'].isin(embedding_subreddits)]
    retained_posts = len(filtered_df)
    removed_posts = total_posts - retained_posts
    
    pct_retained = (retained_posts / total_posts) * 100
    pct_removed = (removed_posts / total_posts) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    categories = ['Enhanced Posts\nDataset (Total)', 'Retained\n(with embeddings)', 'Only in Enhanced\nPosts Dataset']
    values = [total_posts, retained_posts, removed_posts]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Posts/Edges', fontsize=12, fontweight='bold')
    ax1.set_title('Post Coverage by Embeddings', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2 = axes[1]
    labels = [f'Posts retained\n({pct_retained:.1f}%)',
              f'Posts only in enhanced dataset\n({pct_removed:.1f}%)']
    sizes = [retained_posts, removed_posts]
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                                         colors=colors_pie, explode=explode,
                                         startangle=90, textprops={'fontsize': 10})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax2.set_title('Post Retention Rate\n(Source Subreddit Filter)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    fig1 = analyze_subreddit_overlap()
    fig2 = analyze_post_coverage()
    plt.show()
