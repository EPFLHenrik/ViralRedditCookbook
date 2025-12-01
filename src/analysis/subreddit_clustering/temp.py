import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.utils.data_utils import load_reddit_df, load_enhanced_df

def prepare_virality_labels(df, cluster_col):
    df['viral'] = False
    for cluster in df[cluster_col].dropna().unique():
        mask = df[cluster_col] == cluster
        cluster_data = df.loc[mask, 'ups']
        threshold = cluster_data.mean() + 2 * cluster_data.std()
        df.loc[mask & (df['ups'] > threshold), 'viral'] = True
    return df

def feature_engineering(df):
    sentiment_features = [c for c in df.columns if c.startswith('LIWC_') or c.startswith('vader')]
    
    property_features = ['POST_PROPERTIES_num_chars_in_body', 
                        'POST_PROPERTIES_num_digits', 
                        'LINK_SENTIMENT']
    
    return sentiment_features + [f for f in property_features if f in df.columns]

def train_logistic_model(df, cluster_col, features):
    results = {}
    for cluster in df[cluster_col].dropna().unique():
        cluster_df = df[df[cluster_col] == cluster].copy()
        
        X = cluster_df[features].fillna(0)
        y = cluster_df['viral']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        coef_df = pd.DataFrame({
            'feature': features,
            'coefficient': model.coef_[0],
            'abs_coef': np.abs(model.coef_[0])
        }).sort_values('abs_coef', ascending=False)
        
        results[cluster] = {
            'model': model,
            'scaler': scaler,
            'coefficients': coef_df,
            'accuracy': model.score(X_scaled, y)
        }
    
    return results

def compare_viral_vs_nonviral(df, cluster_col, features):
    results = []
    
    for cluster in df[cluster_col].dropna().unique():
        cluster_df = df[df[cluster_col] == cluster]
        
        for feature in features:
            viral = cluster_df[cluster_df['viral']][feature].dropna()
            non_viral = cluster_df[~cluster_df['viral']][feature].dropna()
            
            if len(viral) > 1 and len(non_viral) > 1:
                t, p = stats.ttest_ind(viral, non_viral)
                
                results.append({
                    'cluster': cluster,
                    'feature': feature,
                    'viral_mean': viral.mean(),
                    'non_viral_mean': non_viral.mean(),
                    'mean_diff': viral.mean() - non_viral.mean(),
                    't_stat': t,
                    'p_value': p,
                    'significant': p < 0.05
                })
    
    return pd.DataFrame(results)

def export_virality_results(results_df, model_results, output_dir):
    for cluster, res in model_results.items():
        res['coefficients'].to_csv(
            os.path.join(output_dir, f'cluster_{cluster}_feature_importance.csv'),
            index=False
        )
    
    results_df.to_csv(os.path.join(output_dir, 'viral_vs_nonviral_ttests.csv'), index=False)
    
    print("\nTOP PREDICTIVE FEATURES PER CLUSTER:")
    for cluster, res in model_results.items():
        print(f"\nCluster {cluster} (Accuracy: {res['accuracy']:.2%}):")
        print(res['coefficients'].head(10))

def main():
    df = load_reddit_df(os.path.join('data', "reddit_df_final_superclusters.pkl"))
    df_enh = load_enhanced_df()
    df["ups"] = df_enh["ups"]
    df = df.dropna(subset=['ups'])
    
    tgt_cols = [c for c in df.columns if 'target' in c.lower() and 'cluster' in c.lower()]
    cluster_col = next((c for c in tgt_cols if 'level_1' in c.lower()), tgt_cols[0])
    
    df = prepare_virality_labels(df, cluster_col)
    
    features = feature_engineering(df)
    
    model_results = train_logistic_model(df, cluster_col, features)
    
    ttest_results = compare_viral_vs_nonviral(df, cluster_col, features)
    
    output_dir = 'src/analysis/subreddit_clustering/results/virality'
    os.makedirs(output_dir, exist_ok=True)
    export_virality_results(ttest_results, model_results, output_dir)

if __name__ == "__main__":
    main()
