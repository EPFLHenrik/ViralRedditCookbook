import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from src.utils.data_utils import load_reddit_df, load_enhanced_df

def calculate_cluster_correlations(df, cluster_col, cluster_names, metrics):
    clusters = df[cluster_col].dropna().unique()
    results = []
    
    for metric in metrics:
        for c1, c2 in combinations(sorted(clusters), 2):
            d1 = df[df[cluster_col] == c1][metric].dropna()
            d2 = df[df[cluster_col] == c2][metric].dropna()
            
            if len(d1) > 1 and len(d2) > 1:
                t, p = stats.ttest_ind(d1, d2)
                m1, m2 = d1.mean(), d2.mean()
                
                results.append({
                    'metric': metric,
                    'cluster1': c1,
                    'cluster1_name': cluster_names.get(c1, str(c1)),
                    'cluster2': c2,
                    'cluster2_name': cluster_names.get(c2, str(c2)),
                    'mean1': m1,
                    'mean2': m2,
                    'median1': d1.median(),
                    'median2': d2.median(),
                    'mean_diff': abs(m1 - m2),
                    'mean_ratio': max(m1, m2) / max(min(m1, m2), 0.001),
                    'std1': d1.std(),
                    'std2': d2.std(),
                    'n1': len(d1),
                    'n2': len(d2),
                    't_statistic': t,
                    'p_value': p,
                    'significant': p < 0.05
                })
    
    return pd.DataFrame(results)

def export_results(results_df, output_dir, metrics):
    results_df.to_csv(os.path.join(output_dir, 'cluster_correlations_ttests.csv'), index=False)
    
    print(f"\nTotal comparisons: {len(results_df)}")
    print(f"Significant (p<0.05): {results_df['significant'].sum()} ({100*results_df['significant'].mean():.1f}%)")
    
    sig_results = results_df[results_df['significant']].sort_values('p_value')
    
    print(f"\nMOST SIGNIFICANT PAIRS (Lowest P-Values):")
    for _, r in sig_results.head(20).iterrows():
        print(f"\n{r['cluster1_name']} vs {r['cluster2_name']} ({r['metric']})")
        print(f"  Means: {r['mean1']:.2f} vs {r['mean2']:.2f} (diff: {r['mean_diff']:.2f})")
        print(f"  t-statistic: {r['t_statistic']:.3f}, p-value: {r['p_value']:.6e}")
        print(f"  Sample sizes: n1={r['n1']}, n2={r['n2']}")
    
    print(f"\nLARGEST DIFFERENCES (By Mean Ratio):")
    for _, r in sig_results.sort_values('mean_ratio', ascending=False).head(15).iterrows():
        print(f"\n{r['cluster1_name']} vs {r['cluster2_name']} ({r['metric']})")
        print(f"  Means: {r['mean1']:.2f} vs {r['mean2']:.2f} ({r['mean_ratio']:.2f}x)")
        print(f"  t-statistic: {r['t_statistic']:.3f}, p-value: {r['p_value']:.4e}")
    
    print(f"\nMOST SIGNIFICANT PAIRS PER METRIC:")
    for metric in metrics:
        metric_data = results_df[(results_df['metric'] == metric) & (results_df['significant'])]
        if len(metric_data) > 0:
            print(f"\n{metric.upper()}:")
            for i, (_, r) in enumerate(metric_data.sort_values('p_value').head(5).iterrows(), 1):
                print(f"  {i}. {r['cluster1_name']} vs {r['cluster2_name']}")
                print(f"     Means: {r['mean1']:.2f} vs {r['mean2']:.2f}, p={r['p_value']:.4e}")

def main():
    df = load_reddit_df(os.path.join('data', "reddit_df_final_superclusters.pkl"))
    df_enh = load_enhanced_df()
    
    df["ups"] = df_enh["ups"]
    if "downs" in df_enh.columns:
        df["downs"] = df_enh["downs"]
    if "score" in df_enh.columns:
        df["score"] = df_enh["score"]
    if "num_comments" in df_enh.columns:
        df["num_comments"] = df_enh["num_comments"]
    
    df = df.dropna(subset=['ups'])
    
    output_dir = 'src/analysis/subreddit_clustering/results'
    os.makedirs(output_dir, exist_ok=True)
    
    src_cols = [c for c in df.columns if 'source' in c.lower() and 'cluster' in c.lower()]
    tgt_cols = [c for c in df.columns if 'target' in c.lower() and 'cluster' in c.lower()]
    
    src_col = next((c for c in src_cols if 'level_1' in c.lower()), src_cols[0] if src_cols else None)
    tgt_col = next((c for c in tgt_cols if 'level_1' in c.lower()), tgt_cols[0] if tgt_cols else None)
    
    cluster_col = tgt_col or src_col
    
    if not cluster_col:
        print("Error: No cluster columns found!")
        return
    
    name_col = cluster_col.replace('cluster', 'cluster_name')
    if name_col in df.columns:
        cluster_names = df.groupby(cluster_col)[name_col].first().to_dict()
    else:
        cluster_names = {c: f"Cluster_{c}" for c in df[cluster_col].dropna().unique()}
    
    metrics = [m for m in ['ups', 'downs', 'score', 'num_comments'] 
               if m in df.columns and df[m].notna().sum() > 0]
    
    results = calculate_cluster_correlations(df, cluster_col, cluster_names, metrics)
    export_results(results, output_dir, metrics)

if __name__ == "__main__":
    main()
