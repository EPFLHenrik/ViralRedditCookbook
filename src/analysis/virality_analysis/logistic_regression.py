import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.utils.data_utils import load_final_df
from src.analysis.virality_analysis.get_viral_posts import get_viral_posts


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'logistic_regression')


def logistic_regression(df, save_to_csv=True):
    """
    Train a logistic regression model to predict virality (is_viral) using selected features.
    
    Args:
        df: DataFrame with is_viral column and various features
    """
    
    df_model = df.copy()
    
    df_model['is_viral_numeric'] = df_model['is_viral'].astype(int)
    
    numeric_features_list = [
        'Num_words',
        'Avg_word_length',
        'Automated_readability_index',
        'Sentiment_compound_vader',
        'LIWC_Posemo',
        'LIWC_Negemo',
        'LIWC_Social',
        'LIWC_Affect',
        'LIWC_CogMech',
        'LIWC_Swear',
        'LINK_SENTIMENT',
    ]
    
    print(f"Viral posts: {df_model['is_viral_numeric'].sum()} ({df_model['is_viral_numeric'].mean()*100:.2f}%)")
    print(f"Non-viral posts: {(df_model['is_viral_numeric'] == 0).sum()} ({(df_model['is_viral_numeric'] == 0).mean()*100:.2f}%)")
    
    scaler = StandardScaler()
    df_model_scaled = df_model.copy()
    for col in numeric_features_list:
        if col in df_model_scaled.columns:
            scaled_values = scaler.fit_transform(df_model_scaled[[col]])
            scaled_values = np.clip(scaled_values, -6, 6) # clip to +/- 6 standard deviations (to prevent overflow in logistic regression)
            df_model_scaled[col] = scaled_values

    formula = "is_viral_numeric ~ "
    formula += " + ".join(numeric_features_list)
    formula += " + C(source_supercluster)"
    
    print(f"\nFormula: {formula}\n")
    
    model = smf.logit(formula=formula, data=df_model_scaled).fit(
        maxiter=1000,
    )
    
    print("LOGISTIC REGRESSION RESULTS")
    print(model.summary())
    
    print("MODEL PERFORMANCE METRICS")
    
    predictions = model.predict(df_model_scaled)
    predicted_viral = (predictions >= 0.5).astype(int)
    actual_viral = df_model_scaled['is_viral_numeric'].values
    
    accuracy = accuracy_score(actual_viral, predicted_viral)
    precision = precision_score(actual_viral, predicted_viral, zero_division=0)
    recall = recall_score(actual_viral, predicted_viral, zero_division=0)
    f1 = f1_score(actual_viral, predicted_viral, zero_division=0)
    roc_auc = roc_auc_score(actual_viral, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(actual_viral, predicted_viral)
    print(f"Confusion Matrix: {cm}")
    print(cm)
    
    print("FEATURE COEFFICIENTS (Top 20 by absolute value)")
    
    coef_df = pd.DataFrame({
        'Feature': model.params.index,
        'Coefficient': model.params.values,
        'P-value': model.pvalues.values,
        'Abs_Coefficient': np.abs(model.params.values)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(coef_df.head(20).to_string(index=False))
    
    if save_to_csv:
        coef_no_intercept = coef_df[coef_df['Feature'] != 'Intercept'].copy()
        top_positive = coef_no_intercept[coef_no_intercept['Coefficient'] > 0].nlargest(8, 'Coefficient')
        top_negative = coef_no_intercept[coef_no_intercept['Coefficient'] < 0].nsmallest(8, 'Coefficient')
        
        export_data = []
        
        for _, row in top_positive.iterrows():
            export_data.append({
                'word': row['Feature'],
                'coefficient_up': row['Coefficient'],
                'coefficient_down': ''
            })
        
        for _, row in top_negative.iterrows():
            export_data.append({
                'word': row['Feature'],
                'coefficient_up': '',
                'coefficient_down': row['Coefficient']
            })
        
        export_df = pd.DataFrame(export_data)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        output_path = os.path.join(OUTPUT_DIR, 'viral_properties.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    df = load_final_df(calculate_rss=True)
    df = get_viral_posts(df, virality_percentile=0.999, window_size='60D')
    model, df_model = logistic_regression(df)
