import pandas as pd
import numpy as np
import os

FEATURE_KEEP_PROPORTION_THRESHOLD = 0.05  # Min proportion of significant pairs to keep a feature
SIGNIFICANT_CORR_THRESHOLD = 0.25  # Threshold for significant correlation of feature pairs
VARYING_STD_THRESHOLD = 0.3  # Std threshold for varying feature pairs
HIGH_CORR_PAIR_THRESHOLD = 0.7  # Threshold for prop of high correlation subreddits (a feature pair correlated across this proportion of cases)
LOW_CORR_THRESHOLD = 0.1  # Threshold for prop of low feature correlations between subreddit pairs
INTERESTING_RANGE_THRESHOLD = 0.5  # Range threshold for 'interesting' feature pairs (i.e. if max - min > threshold across occurrences)

def analyze_pair_matrices(csv_path, metric_name, output_dir, low_threshold=FEATURE_KEEP_PROPORTION_THRESHOLD, high_threshold=SIGNIFICANT_CORR_THRESHOLD):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Unique subreddit pairs: {df[['sub1', 'sub2']].drop_duplicates().shape[0]}")
    print(f"Unique feature pairs: {df[['feature1', 'feature2']].drop_duplicates().shape[0]}")
    
    stats = df.groupby(['feature1', 'feature2'])['value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    stats['abs_max'] = stats['max'].abs()
    stats = stats.sort_values('abs_max', ascending=False)
    stats.to_csv(f"{output_dir}/feature_pair_stats_{metric_name}.csv", index=False)
    
    significant_pairs = stats[stats['abs_max'] > high_threshold]
    print(f"Significant feature pairs (|max {metric_name}| > {high_threshold}): {len(significant_pairs)}")
    significant_pairs.to_csv(f"{output_dir}/significant_feature_pairs_{metric_name}.csv", index=False)
    
    varying_pairs = stats[stats['std'] > VARYING_STD_THRESHOLD]
    print(f"Varying feature pairs (std > {VARYING_STD_THRESHOLD}): {len(varying_pairs)}")
    varying_pairs.to_csv(f"{output_dir}/varying_feature_pairs_{metric_name}.csv", index=False)
    
    high_corr_per_pair = df[df['value'].abs() > HIGH_CORR_PAIR_THRESHOLD]
    high_corr_per_pair.to_csv(f"{output_dir}/high_{metric_name}_per_pair.csv", index=False)
    
    insignificant_features = stats[stats['abs_max'] < LOW_CORR_THRESHOLD]
    print(f"Insignificant features (max abs < {LOW_CORR_THRESHOLD}): {insignificant_features['feature1'].unique()}")
    
    interesting_pairs = stats[(stats['max'] - stats['min']) > INTERESTING_RANGE_THRESHOLD]
    interesting_pairs.to_csv(f"{output_dir}/interesting_varying_pairs_{metric_name}.csv", index=False)
    print(f"Interesting varying pairs (max - min > {INTERESTING_RANGE_THRESHOLD}): {len(interesting_pairs)}")
    
    for _, row in interesting_pairs.iterrows():
        f1, f2 = row['feature1'], row['feature2']
        sub_df = df[(df['feature1'] == f1) & (df['feature2'] == f2)]
        high_subs = sub_df[sub_df['value'] > high_threshold][['sub1', 'sub2', 'value']]
        low_subs = sub_df[sub_df['value'] < LOW_CORR_THRESHOLD][['sub1', 'sub2', 'value']]
        if not high_subs.empty and not low_subs.empty:
            print(f"Feature pair {f1} - {f2}: High in {len(high_subs)} pairs, Low in {len(low_subs)} pairs")
    features_to_remove = []
    all_features = set(df['feature1'].unique()) | set(df['feature2'].unique())
    for feat in all_features:
        feat_pairs = df[(df['feature1'] == feat) | (df['feature2'] == feat)]
        significant_count = (feat_pairs['value'].abs() > high_threshold).sum()
        total_pairs = len(feat_pairs)
        if total_pairs > 0 and (significant_count / total_pairs) < low_threshold:
            features_to_remove.append(feat)
    features_to_keep = list(all_features - set(features_to_remove))
    print(f"Features to keep (significant in >= {low_threshold} of pairs): {features_to_keep}")
    with open(f"{output_dir}/features_to_keep_{metric_name}.txt", 'w') as f:
        f.write('\n'.join(features_to_keep))
    return features_to_keep, interesting_pairs

if __name__ == "__main__":
    metric = 'corr'
    output_dir = 'src/analysis/top_subreddits_analysis_charts/liwc_corr_cov/analysis'
    base_dir = 'src/analysis/top_subreddits_analysis_charts/liwc_corr_cov'
    csv_path = f"{base_dir}/all_{metric}.csv"
    
    features_to_keep, interesting_pairs = analyze_pair_matrices(csv_path, metric, output_dir)
    print(f"Interesting pairs ({len(interesting_pairs)}):")
    for _, row in interesting_pairs.iterrows():
        print(f"  {row['feature1']} - {row['feature2']}: range {row['max'] - row['min']:.3f}")
