import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.utils.data_utils import load_reddit_graph, load_reddit_df

def find_zeros(df):
    feature_columns = [col for col in df.columns if col not in ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 'TIMESTAMP']]
    
    for col in feature_columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    zero_percentages = []
    feature_names = []
    
    for col in feature_columns:
        if df[col].dtype in ['float64', 'int64']:
            zero_count = ((df[col] == 0) | df[col].isna()).sum()
            total_count = len(df)
            zero_percentage = (zero_count / total_count) * 100 if total_count > 0 else 0
            zero_percentages.append(zero_percentage)
            feature_names.append(col)
    
    plt.figure(figsize=(20, 8))
    bars = plt.bar(range(len(feature_names)), zero_percentages)
    plt.xlabel('Features')
    plt.ylabel('Percentage of Zero Values')
    plt.title('Percentage of Zero Values for Each Feature')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.tight_layout()
    
    import os
    output_dir = 'figures/init_analysis_charts'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/zero_percentages.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return dict(zip(feature_names, zero_percentages))

def main():
    df = load_reddit_df()
    
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"Sample of first few rows:")
    print(df.head())
    
    zero_stats = find_zeros(df)
    
    print(f"\nTop 10 features with highest zero percentages:")
    sorted_stats = sorted(zero_stats.items(), key=lambda x: x[1], reverse=True)
    for feature, percentage in sorted_stats[:10]:
        print(f"{feature}: {percentage:.2f}%")

if __name__ == "__main__":
    main()