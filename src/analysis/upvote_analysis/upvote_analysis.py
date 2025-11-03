import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.data_utils import load_enhanced_df

def upvote_distribution_by_subreddit():
    df = load_enhanced_df()

    df.loc[df["LINK_SENTIMENT"] == -1, "LINK_SENTIMENT"] = 0
    df["LINK_SENTIMENT"] = df["LINK_SENTIMENT"].astype(int)
    df = df.dropna(subset=['ups'])

    percentiles = (
        df.groupby('SOURCE_SUBREDDIT')['ups']
        .agg(
            minimum='min',
            q25=lambda x: np.percentile(x, 25),
            mean='mean',
            median='median',
            q75=lambda x: np.percentile(x, 75),
            count='count',
            maximum='max'
        )
        .reset_index()
    )

    percentiles = percentiles[percentiles["count"] > 5]

    max_val = percentiles['mean'].max()
    bins = 10 ** np.arange(0, np.ceil(np.log10(max_val)) + 1)
    labels = [f"{int(bins[i]):,}–{int(bins[i+1]):,}" for i in range(len(bins)-1)]

    percentiles['mean_ups_group'] = pd.cut(percentiles['mean'], bins=bins, labels=labels, right=False)
    ncols = 2
    nrows = 2
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 20, nrows * 15), squeeze=False)
    for idx, l in enumerate(labels[:4]):
        ax = axes[idx // ncols][idx % ncols]
        l_subs = percentiles[percentiles['mean_ups_group'] == l]
        random_ones = l_subs.sample(n=min(20, l_subs.shape[0]), random_state=42) if l_subs.shape[0] > 0 else l_subs
        if random_ones.shape[0] == 0:
            ax.text(0.5, 0.5, 'No subreddits', ha='center', va='center')
            ax.set_title(str(l), fontsize=14)
            continue
        for i, row in enumerate(random_ones.itertuples()):
            ax.plot([row.q25, row.q75], [i, i], color='steelblue', linewidth=(np.log2(row.count)+1)*2, solid_capstyle='round')
            ax.scatter(row.mean, i, color='yellow', zorder=3, s=200, label='Mean' if i == 0 else '')
            ax.scatter(row.minimum, i, color='red', zorder=3, s=200, label='Min' if i == 0 else '')
            ax.scatter(row.maximum, i, color='green', zorder=3, s=200, label='Max' if i == 0 else '')
            ax.scatter(row.median, i, color='black', zorder=3, s=200, label='Median' if i == 0 else '')
        ax.set_yticks(range(len(random_ones)))
        ax.set_yticklabels(random_ones['SOURCE_SUBREDDIT'], fontsize=30)
        ax.set_xlabel('Upvotes', fontsize=25)
        ax.tick_params(axis='x', labelsize=30)
        ax.set_title(f'Upvote Distribution (25th–75th Percentile) \nfor Subreddits with mean {l}', fontsize=40)
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', alpha=0.4)
        if random_ones.shape[0] > 0:
            ax.legend(fontsize=25)
    plt.tight_layout()
    plt.show()

def upvote_distribution():
    df = load_enhanced_df()

    df.loc[df["LINK_SENTIMENT"] == -1, "LINK_SENTIMENT"] = 0
    df["LINK_SENTIMENT"] = df["LINK_SENTIMENT"].astype(int)
    df = df.dropna(subset=['ups'])
    df['mean_ups']  = df.groupby('SOURCE_SUBREDDIT')['ups'].transform('mean') 
    df['post_count'] = df.groupby('SOURCE_SUBREDDIT')['ups'].transform('count') 


    df['rel_ups'] = df['ups'] / df['mean_ups']

    df['log_rel'] = np.log10(df['rel_ups'])

    max_val = df['mean_ups'].max()
    bins = 10 ** np.arange(0, np.ceil(np.log10(max_val)) + 1)
    labels = [f"{int(bins[i]):,}–{int(bins[i+1]):,}" for i in range(len(bins)-1)]

    df['mean_ups_group'] = pd.cut(df['mean_ups'], bins=bins, labels=labels, right=False)


    ncols = 3
    nrows = 2
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), squeeze=False)
    for idx, l in enumerate(labels):
        ax = axes[idx // ncols][idx % ncols]
        data = df[df['mean_ups_group'] == l]
        if data.shape[0] == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(str(l), fontsize=14)
            continue
        sns.kdeplot(data=data, x='log_rel', fill=True, color='skyblue', ax=ax)
        ax.axvline(0, color='red', linestyle='--', label='Mean (ratio = 1)')
        ax.set_title(f'Distribution (mean upvotes {l})', fontsize=14)
        ax.set_xlabel('log10(Relative Upvotes)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

def see_average_upvotes():
    df = load_enhanced_df()

    df.loc[df["LINK_SENTIMENT"] == -1, "LINK_SENTIMENT"] = 0
    df["LINK_SENTIMENT"] = df["LINK_SENTIMENT"].astype(int)
    df = df.dropna(subset=['ups'])
    mean_by_sub = (
        df.groupby('SOURCE_SUBREDDIT', as_index=False)['ups']
        .mean()
        .rename(columns={'ups': 'mean_ups'})
    )

    post_counts = (
        df.groupby('SOURCE_SUBREDDIT')
        .size()
        .reset_index(name='post_count')
    )

    ups_summary = (
        mean_by_sub
        .merge(post_counts, on='SOURCE_SUBREDDIT')
        .sort_values('mean_ups', ascending=False)
    )

    ups_nonzero = ups_summary[ups_summary['mean_ups'] > 0].copy()
    ups_nonzero['log_mean_ups'] = np.log10(ups_nonzero['mean_ups'])

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=ups_nonzero,
        x='log_mean_ups',
        bins=50,
        kde=True,
        color='steelblue'
    )

    plt.title('Log-Scaled Distribution of Average Ups per Subreddit')
    plt.xlabel('log10(Average Ups per Subreddit)')
    plt.ylabel('Frequency (Number of Subreddits)')
    
    mean_log = ups_nonzero['log_mean_ups'].mean()
    std_log = ups_nonzero['log_mean_ups'].std()
    plt.axvline(mean_log, color='red', linestyle='--', label=f'Mean log = {mean_log:.2f}')
    plt.axvline(mean_log + std_log, color='green', linestyle=':', label='+1 SD')
    plt.axvline(mean_log - std_log, color='green', linestyle=':', label='-1 SD')
    plt.legend()

    plt.tight_layout()
    plt.show()