import pandas as pd 
import sys, os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.data_utils import load_enhanced_df, load_reddit_df

def upvote_distribution_by_subreddit():
    """
    Analyze and visualize upvote distributions by subreddit, grouping subreddits by mean upvote ranges.
    """
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
    plt.tight_layout()
    plt.show()

def upvote_distribution_by_subreddit_one_range():
    """
    Analyze and visualize upvote distributions for subreddits in a specific mean upvote range.
    """
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
    _, axes = plt.subplots(1, 1, figsize=(40, 30), squeeze=False)
    
    l = labels[2]
    l_subs = percentiles[percentiles['mean_ups_group'] == l]

    random_ones = l_subs.sample(n=min(20, l_subs.shape[0]), random_state=42) if l_subs.shape[0] > 0 else l_subs
    ax = axes[0][0]
    for i, row in enumerate(random_ones.itertuples()):
        boxplot_stats = [{
            'whislo': row.minimum,
            'q1': row.q25,
            'med': row.median,
            'q3': row.q75,
            'whishi': row.maximum,
            'fliers': [],
            'mean': row.mean,
        }]
        ax.bxp(boxplot_stats,
            positions=[i],
            vert=False,
            patch_artist=True,
            showfliers=False,
            boxprops={'facecolor': '#B3C3F3', 'edgecolor': 'black'},
            widths=0.6,
            medianprops={'color': 'red', 'linewidth': 2}
        )
    ax.set_yticks(range(len(random_ones)))
    ax.set_yticklabels(random_ones['SOURCE_SUBREDDIT'], fontsize=30)
    ax.set_xlabel('Upvotes', fontsize=25)
    ax.tick_params(axis='x', labelsize=30)
    ax.set_title(f'Upvote Distribution (25th–75th Percentile) \nfor Subreddits with mean {l}', fontsize=40)
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()

def upvote_distribution():
    """
    Analyze and visualize relative upvote distributions grouped by subreddit mean upvote ranges.
    """
    df = load_enhanced_df()

    df.loc[df["LINK_SENTIMENT"] == -1, "LINK_SENTIMENT"] = 0
    df["LINK_SENTIMENT"] = df["LINK_SENTIMENT"].astype(int)
    df = df.dropna(subset=['ups'])
    df['mean_ups']  = df.groupby('SOURCE_SUBREDDIT')['ups'].transform('mean') 
    df['post_count'] = df.groupby('SOURCE_SUBREDDIT')['ups'].transform('count') 


    df['rel_ups'] = df['ups'] / df['mean_ups']
    valid_mask = (df['rel_ups'] > 0) & np.isfinite(df['rel_ups'])
    df = df[valid_mask].copy()
    df['log_rel'] = np.log10(df['rel_ups'])

    max_val = df['mean_ups'].max()
    if max_val > 0:
        bins = 10 ** np.arange(0, np.ceil(np.log10(max_val)) + 1)
    else:
        bins = np.array([1, 10, 100, 1000])
    labels = [f"{int(bins[i]):,}–{int(bins[i+1]):,}" for i in range(len(bins)-1)]

    df['mean_ups_group'] = pd.cut(df['mean_ups'], bins=bins, labels=labels, right=False)


    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4), squeeze=False)
    for idx, l in enumerate(labels[:5]):
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
    fig.delaxes(axes[1][2])
    plt.tight_layout()
    plt.show()

def see_average_upvotes():
    """
    Visualize the distribution of average upvotes per subreddit on a log scale.
    """
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


def cluster_upvote_distribution():
    """
    Analyze and visualize upvote distributions by source and target cluster levels.
    """
    DATA_FOLDER = os.path.join(os.getcwd(), 'data')
    df = load_reddit_df(os.path.join(DATA_FOLDER, "reddit_df_merged.pkl"))

    df_enh = load_enhanced_df()

    df["ups"] = df_enh["ups"]
    df = df.dropna(subset=['ups'])

    percentiles_src = (
        df.groupby('source_cluster_level_1')['ups']
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

    percentiles_tgt = (
        df.groupby('target_cluster_level_1')['ups']
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

    percentiles_lst = [percentiles_src, percentiles_tgt]

    _, axes = plt.subplots(2, 1, figsize=(30, 40), squeeze=False)

    for i, percentiles in enumerate(percentiles_lst):
        ax = axes[i // 1][i % 1]
        for i, row in enumerate(percentiles.itertuples()):
            boxplot_stats = [{
                'whislo': row.minimum,
                'q1': row.q25,
                'med': row.median,
                'q3': row.q75,
                'whishi': row.maximum,
                'fliers': [],
                'mean': row.mean,
            }]
            ax.bxp(boxplot_stats,
                positions=[i],
                vert=False,
                patch_artist=True,
                showfliers=False,
                boxprops={'facecolor': '#B3C3F3', 'edgecolor': 'black'},
                widths=0.6,
                medianprops={'color': 'red', 'linewidth': 2}
            )
        ax.set_yticks(range(len(percentiles)))
        ax.set_yticklabels(percentiles.iloc[: ,0], fontsize=20)
        ax.set_xlabel('Upvotes', fontsize=25)
        ax.tick_params(axis='x', labelsize=30)
        ax.set_title(f'Upvote Distribution (25th–75th Percentile) \nfor {percentiles.columns[0]}', fontsize=40)
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.show()

def cluster_percentiles_distribution():
    """
    Analyze and visualize upvote percentile distributions across multiple cluster levels (source and target, levels 1-4).
    """
    DATA_FOLDER = os.path.join(os.getcwd(), 'data')
    df = load_reddit_df(os.path.join(DATA_FOLDER, "reddit_df_merged.pkl"))

    df_enh = load_enhanced_df()

    df["ups"] = df_enh["ups"]
    df = df.dropna(subset=['ups'])

    cluster_groups = ["source_cluster_level_1", "target_cluster_level_1", 
                  "source_cluster_level_2", "target_cluster_level_2",
                  "source_cluster_level_3", "target_cluster_level_3",
                  "source_cluster_level_4", "target_cluster_level_4"]
                  

    ncols = 2
    nrows = 4
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 25, nrows * 20), squeeze=False)

    for i, group in enumerate(cluster_groups):
        ax = axes[i // ncols][i % ncols]
        percentiles = (
            df.groupby(group)['ups']
            .agg(
                minimum='min',
                q10=lambda x: np.percentile(x, 10),
                q20=lambda x: np.percentile(x, 20),
                q30=lambda x: np.percentile(x, 30),
                q40=lambda x: np.percentile(x, 40),
                q50=lambda x: np.percentile(x, 50),
                q60=lambda x: np.percentile(x, 60),
                q70=lambda x: np.percentile(x, 70),
                q80=lambda x: np.percentile(x, 80),
                q90=lambda x: np.percentile(x, 90),
                maximum='max'
            )
            .reset_index()
        )

        cnt = 0
        used_cols = []
        for col_name, col_data in percentiles.items():
            if isinstance(col_data[0], str):
                continue
            used_cols.append(col_name)
            boxplot_stats = [{
                'whislo': col_data.min(),
                'q1': col_data.quantile(0.25),
                'med': col_data.median(),
                'q3': col_data.quantile(0.75),
                'whishi': col_data.max(),
                'fliers': [],
                'mean': col_data.mean(),
            }]
            ax.bxp(boxplot_stats,
                positions=[cnt],
                vert=False,
                patch_artist=True,
                showfliers=False,
                boxprops={'facecolor': '#B3C3F3', 'edgecolor': 'black'},
                widths=0.6,
                medianprops={'color': 'red', 'linewidth': 2}
            )
            cnt += 1

        ax.set_xlabel('Upvotes', fontsize=25)
        ax.set_yticks(range(len(used_cols)))
        ax.set_yticklabels(used_cols, fontsize=20)
        ax.tick_params(axis='x', labelsize=30)
        ax.set_title(f'Upvote Distribution (25th–75th Percentile) \nfor {group}', fontsize=40)
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', alpha=0.4)
    plt.show()