from platform import node
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.utils.data_utils import load_reddit_graph


LIWC_FEATURE_NAMES = [
    "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", 
    "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", 
    "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", 
    "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", 
    "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", 
    "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", 
    "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", 
    "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", 
    "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", 
    "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", 
    "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu"
]

CONTEXTUAL_LIWC_FEATURES = [
    "LIWC_Social", "LIWC_Family", "LIWC_Friends", "LIWC_Humans",
    "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad",
    "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", 
    "LIWC_Certain", "LIWC_Inhib",
    "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel",
    "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest",
    "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time",
    "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig",
    "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu"
]

def find_highest_node_degrees(G):
    """
    Isolate nodes with highest in-degree and out-degree
    """
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    top_in_degree = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    top_out_degree = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 nodes with highest IN-DEGREE:")
    print("Node (Subreddit)\t\tIn-Degree")
    print("-" * 50)
    for node, degree in top_in_degree[:10]:
        print(f"{node:<30}\t{degree}")
    
    print(f"\nTop 10 nodes with highest OUT-DEGREE:")
    print("Node (Subreddit)\t\tOut-Degree")
    print("-" * 50)
    for node, degree in top_out_degree[:10]:
        print(f"{node:<30}\t{degree}")
    
    return top_in_degree, top_out_degree


def analyze_edge_patterns(G, percentile=90):
    """
    Analyze edges connecting nodes with high degrees
    """

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_degree_thresh = np.percentile(list(in_degrees.values()), percentile)
    out_degree_thresh = np.percentile(list(out_degrees.values()), percentile)

    print(f"In-degree threshold for {percentile}th percentile: {in_degree_thresh}")
    print(f"Out-degree threshold for {percentile}th percentile: {out_degree_thresh}")

    high_degree_edges = []

    for src, tgt, data in G.edges(data=True):
        src_out_degree = out_degrees[src]
        tgt_in_degree = in_degrees[tgt]

        if src_out_degree >= out_degree_thresh and tgt_in_degree >= in_degree_thresh:
            high_degree_edges.append({
                'source': src,
                'target': tgt,
                'source_out_degree': src_out_degree,
                'target_in_degree': tgt_in_degree,
                'sentiment': data.get('sentiment', 'Unknown'),
                'timestamp': data.get('timestamp', 'Unknown')
            })
    
    print(f"\nFound {len(high_degree_edges)} edges connecting high-degree nodes...")
    
    # same style dataframe as high_degree_edges but with only unique (source, target) pairs
    high_degree_edges_unique = []
    
    if high_degree_edges:
        high_degree_edges.sort(key=lambda x: x['source_out_degree'] + x['target_in_degree'], reverse=True)
        seen_pairs = set()
        for edge in high_degree_edges:
            pair = (edge['source'], edge['target'])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                high_degree_edges_unique.append(edge)

        print(f"\nFound {len(high_degree_edges_unique)} unique source-target edges connecting high-degree nodes...")

    high_degree_edges = pd.DataFrame(high_degree_edges)
    high_degree_edges_unique = pd.DataFrame(high_degree_edges_unique)

    print("\n\n\nHigh-degree edges:")
    print(high_degree_edges.head(10))
    print("\n\n\nWithout Duplicates:")
    print(high_degree_edges_unique.head(10))

    return high_degree_edges, high_degree_edges_unique


def extract_liwc_features(G, top_subreddits=None, features=CONTEXTUAL_LIWC_FEATURES):
    top_set = set(top_subreddits) if top_subreddits is not None else None
    rows = []
    for src, dst, edge_data in G.edges(data=True):
        if top_set is not None and src not in top_set:
            continue
        props = edge_data.get('properties')
        if props is None or len(props) < 85:
            continue
        liwc_vals = props[21:85]
        row = {'subreddit': src}
        row.update(dict(zip(LIWC_FEATURE_NAMES, liwc_vals)))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[['subreddit'] + features]


def calculate_liwc_covariance_matrix(liwc_df, output_dir, features=CONTEXTUAL_LIWC_FEATURES, show=False):
    """
    Calculate LIWC feature covariance matrix and export heatmaps
    """
    os.makedirs(output_dir, exist_ok=True)

    cov_mat = liwc_df[features].cov()
    corr_mat = liwc_df[features].corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(cov_mat, cmap='RdBu_r', center=0, 
                ax=axes[0], cbar_kws={'label': 'Covariance'}, 
                xticklabels=[f.replace('LIWC_', '') for f in cov_mat.columns],
                yticklabels=[f.replace('LIWC_', '') for f in cov_mat.index])
    sns.heatmap(corr_mat, cmap='RdBu_r', center=0, 
                ax=axes[1], cbar_kws={'label': 'Correlation'},
                xticklabels=[f.replace('LIWC_', '') for f in corr_mat.columns],
                yticklabels=[f.replace('LIWC_', '') for f in corr_mat.index])
    
    axes[0].set_title('LIWC Feature Covariance Matrix')
    axes[1].set_title('LIWC Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/liwc_feature_cov.png", dpi=300, bbox_inches='tight')
    if(show):
        plt.show()
    plt.close()
    
    return cov_mat, corr_mat


def calculate_subreddit_similarity_matrix(liwc_df, output_dir, features=CONTEXTUAL_LIWC_FEATURES, show=False):
    os.makedirs(f"{output_dir}/liwc_corr_cov", exist_ok=True)
    subs = liwc_df['subreddit'].unique()
    cov_dict = {}
    corr_dict = {}
    cov_rows = []
    corr_rows = []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            sub1, sub2 = subs[i], subs[j]
            df1 = liwc_df[liwc_df['subreddit'] == sub1][features]
            df2 = liwc_df[liwc_df['subreddit'] == sub2][features]
            combined = pd.concat([df1, df2], ignore_index=True).dropna()
            if combined.shape[0] < 2:
                continue
            cov_mat = combined.cov()
            corr_mat = combined.corr()
            cov_dict[(sub1, sub2)] = cov_mat
            corr_dict[(sub1, sub2)] = corr_mat
            for fi in features:
                for fj in features:
                    cov_rows.append([sub1, sub2, fi, fj, cov_mat.loc[fi, fj]])
                    corr_rows.append([sub1, sub2, fi, fj, corr_mat.loc[fi, fj]])
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            sns.heatmap(cov_mat, cmap='RdBu_r', center=0, ax=axes[0], cbar_kws={'label': 'Covariance'},
                        xticklabels=[f.replace('LIWC_', '') for f in cov_mat.columns],
                        yticklabels=[f.replace('LIWC_', '') for f in cov_mat.index])
            axes[0].set_title(f'Covariance: {sub1} vs {sub2}')
            sns.heatmap(corr_mat, cmap='RdBu_r', center=0, ax=axes[1], cbar_kws={'label': 'Correlation'},
                        xticklabels=[f.replace('LIWC_', '') for f in corr_mat.columns],
                        yticklabels=[f.replace('LIWC_', '') for f in corr_mat.index])
            axes[1].set_title(f'Correlation: {sub1} vs {sub2}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/liwc_corr_cov/cov_corr_{sub1}_{sub2}.png", dpi=300, bbox_inches='tight')
            plt.close()
    df_cov = pd.DataFrame(cov_rows, columns=['sub1', 'sub2', 'feature1', 'feature2', 'value'])
    df_corr = pd.DataFrame(corr_rows, columns=['sub1', 'sub2', 'feature1', 'feature2', 'value'])
    df_cov.to_csv(f"{output_dir}/liwc_corr_cov/all_cov.csv", index=False)
    df_corr.to_csv(f"{output_dir}/liwc_corr_cov/all_corr.csv", index=False)
    return cov_dict, corr_dict


def main(features=CONTEXTUAL_LIWC_FEATURES, output_dir="charts"):
    G = load_reddit_graph()
    top_in_degree, top_out_degree = find_highest_node_degrees(G)
    high_degree_edges, high_degree_edges_unique = analyze_edge_patterns(G)
    top_subreddits = [node for node, _ in top_in_degree[:20]] + [node for node, _ in top_out_degree[:20]]
    top_subreddits = list(set(top_subreddits))
    liwc_df = extract_liwc_features(G, top_subreddits, features)
    lwic_cov, lwic_corr = calculate_liwc_covariance_matrix(liwc_df, output_dir, features)
    cov_dict, corr_dict = calculate_subreddit_similarity_matrix(liwc_df, output_dir, features)


if __name__ == "__main__":
    main()