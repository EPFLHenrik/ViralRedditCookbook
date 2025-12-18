import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
from src.utils.data_utils import load_final_df

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
                    'cluster1': cluster_names.get(c1, str(c1)),
                    'cluster2': cluster_names.get(c2, str(c2)),
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
        print(f"\n{r['cluster1']} vs {r['cluster2']} ({r['metric']})")
        print(f"  Means: {r['mean1']:.2f} vs {r['mean2']:.2f} (diff: {r['mean_diff']:.2f})")
        print(f"  t-statistic: {r['t_statistic']:.3f}, p-value: {r['p_value']:.6e}")
        print(f"  Sample sizes: n1={r['n1']}, n2={r['n2']}")
    
    print(f"\nLARGEST DIFFERENCES (By Mean Ratio):")
    for _, r in sig_results.sort_values('mean_ratio', ascending=False).head(15).iterrows():
        print(f"\n{r['cluster1']} vs {r['cluster2']} ({r['metric']})")
        print(f"  Means: {r['mean1']:.2f} vs {r['mean2']:.2f} ({r['mean_ratio']:.2f}x)")
        print(f"  t-statistic: {r['t_statistic']:.3f}, p-value: {r['p_value']:.4e}")
    
    print(f"\nMOST SIGNIFICANT PAIRS PER METRIC:")
    for metric in metrics:
        metric_data = results_df[(results_df['metric'] == metric) & (results_df['significant'])]
        if len(metric_data) > 0:
            print(f"\n{metric.upper()}:")
            for i, (_, r) in enumerate(metric_data.sort_values('p_value').head(5).iterrows(), 1):
                print(f"  {i}. {r['cluster1']} vs {r['cluster2']}")
                print(f"     Means: {r['mean1']:.2f} vs {r['mean2']:.2f}, p={r['p_value']:.4e}")

def export_cluster_means(df, cluster_col, cluster_names, metric, output_dir):
    cluster_stats = []
    
    for cluster in sorted(df[cluster_col].dropna().unique()):
        cluster_data = df[df[cluster_col] == cluster][metric].dropna()
        
        if len(cluster_data) > 0:
            cluster_stats.append({
                'cluster': cluster_names.get(cluster, str(cluster)),
                'mean': cluster_data.mean(),
                'median': cluster_data.median(),
                'std': cluster_data.std(),
                'n': len(cluster_data)
            })
    
    stats_df = pd.DataFrame(cluster_stats).sort_values('mean', ascending=True)
    stats_df.to_csv(os.path.join(output_dir, f'{metric}_by_cluster.csv'), index=False)
    
    print(f"\n\nCLUSTERS RANKED BY {metric.upper()}:")
    for i, row in stats_df.iterrows():
        print(f"{row['cluster']}: mean={row['mean']:.4f}, median={row['median']:.4f}, n={row['n']}")
    
    return stats_df

def export_pvalue_matrix(df, cluster_col, cluster_names, metric, output_dir):
    clusters = sorted(df[cluster_col].dropna().unique())
    cluster_labels = [cluster_names.get(c, str(c)) for c in clusters]
    
    pvalue_matrix = pd.DataFrame(index=cluster_labels, columns=cluster_labels, dtype=float)
    
    for c1 in clusters:
        for c2 in clusters:
            if c1 == c2:
                pvalue_matrix.loc[cluster_names.get(c1, str(c1)), cluster_names.get(c2, str(c2))] = 1.0
            else:
                d1 = df[df[cluster_col] == c1][metric].dropna()
                d2 = df[df[cluster_col] == c2][metric].dropna()
                
                if len(d1) > 1 and len(d2) > 1:
                    _, p = stats.ttest_ind(d1, d2)
                    pvalue_matrix.loc[cluster_names.get(c1, str(c1)), cluster_names.get(c2, str(c2))] = p
                else:
                    pvalue_matrix.loc[cluster_names.get(c1, str(c1)), cluster_names.get(c2, str(c2))] = np.nan
    
    pvalue_matrix.to_csv(os.path.join(output_dir, f'{metric}_pvalue_matrix.csv'), float_format='%.10f')
    
    print(f"\n\nP-VALUE MATRIX exported to {metric}_pvalue_matrix.csv")
    
    return pvalue_matrix

def plot_pvalue_heatmap(pvalue_matrix, metric, output_dir):
    plt.figure(figsize=(10, 8))
    
    log_pvalues = -np.log10(pvalue_matrix.replace(0, 1e-300))
    
    mask = pvalue_matrix.isna()
    
    annot_matrix = pvalue_matrix.applymap(lambda x: f'{x:.2e}' if pd.notna(x) else '')
    
    sns.heatmap(log_pvalues, 
                annot=annot_matrix, 
                fmt='',
                cbar_kws={'label': '-log10(p-value)'},
                mask=mask,
                square=True,
                linewidths=0.5,
                annot_kws={'fontsize': 8},
                cmap='Set3'
               )

    plt.title(f'P-value Matrix for {metric}\n(T-tests between clusters, log scale)', fontsize=14, pad=20)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{metric}_pvalue_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"P-VALUE HEATMAP exported to {metric}_pvalue_heatmap.png")


def main():
    df = load_final_df()
    
    output_dir = 'figures/cluster_analysis'
    os.makedirs(output_dir, exist_ok=True)
    cluster_col = 'source_supercluster'
    
    name_col = cluster_col.replace('cluster', 'cluster_name')
    if name_col in df.columns:
        cluster_names = df.groupby(cluster_col)[name_col].first().to_dict()
    else:
        cluster_names = {c: f"Cluster_{c}" for c in df[cluster_col].dropna().unique()}

    metrics = [m for m in ['virality_rss']
            if m in df.columns and df[m].notna().sum() > 0]
    
    results = calculate_cluster_correlations(df, cluster_col, cluster_names, metrics)
    export_results(results, output_dir, metrics)
    
    for metric in metrics:
        export_cluster_means(df, cluster_col, cluster_names, metric, output_dir)
        pvalue_matrix = export_pvalue_matrix(df, cluster_col, cluster_names, metric, output_dir)
        plot_pvalue_heatmap(pvalue_matrix, metric, output_dir)

if __name__ == "__main__":
    main()
