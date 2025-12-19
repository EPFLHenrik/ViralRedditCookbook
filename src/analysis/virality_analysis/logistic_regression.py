import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.utils.data_utils import load_final_df
from src.analysis.virality_analysis.get_viral_posts import get_viral_posts


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'logistic_regression')

NUMERIC_FEATURES_LIST = [
        'Num_characters',
        'Num_characters_no_space',
        'Frac_alpha',
        'Frac_digits',
        'Frac_uppercase',
        'Frac_whitespace',
        'Frac_special',
        'Num_words',
        'Num_unique_words',
        'Num_long_words',
        'Avg_word_length',
        'Num_unique_stopwords',
        'Frac_stopwords',
        'Num_sentences',
        'Num_long_sentences',
        'Avg_chars_per_sentence',
        'Avg_words_per_sentence',
        'Automated_readability_index',
        
        'Sentiment_pos_vader',
        'Sentiment_neg_vader',
        'Sentiment_compound_vader',
        
        'LIWC_Funct',
        'LIWC_Pronoun',
        'LIWC_Ppron',
        'LIWC_I',
        'LIWC_We',
        'LIWC_You',
        'LIWC_SheHe',
        'LIWC_They',
        'LIWC_Ipron',
        'LIWC_Article',
        'LIWC_Verbs',
        'LIWC_AuxVb',
        'LIWC_Past',
        'LIWC_Present',
        'LIWC_Future',
        'LIWC_Adverbs',
        'LIWC_Prep',
        'LIWC_Conj',
        'LIWC_Negate',
        'LIWC_Quant',
        'LIWC_Numbers',
        'LIWC_Swear',
        'LIWC_Social',
        'LIWC_Family',
        'LIWC_Friends',
        'LIWC_Humans',
        'LIWC_Affect',
        'LIWC_Posemo',
        'LIWC_Negemo',
        'LIWC_Anx',
        'LIWC_Anger',
        'LIWC_Sad',
        'LIWC_CogMech',
        'LIWC_Insight',
        'LIWC_Cause',
        'LIWC_Discrep',
        'LIWC_Tentat',
        'LIWC_Certain',
        'LIWC_Inhib',
        'LIWC_Incl',
        'LIWC_Excl',
        'LIWC_Percept',
        'LIWC_See',
        'LIWC_Hear',
        'LIWC_Feel',
        'LIWC_Bio',
        'LIWC_Body',
        'LIWC_Health',
        'LIWC_Sexual',
        'LIWC_Ingest',
        'LIWC_Relativ',
        'LIWC_Motion',
        'LIWC_Space',
        'LIWC_Time',
        'LIWC_Work',
        'LIWC_Achiev',
        'LIWC_Leisure',
        'LIWC_Home',
        'LIWC_Money',
        'LIWC_Relig',
        'LIWC_Death',
        'LIWC_Assent',
        'LIWC_Dissent',
        'LIWC_Nonflu',
        'LIWC_Filler',
        
        'LINK_SENTIMENT',

        'num_body_links',
        'num_title_links',
    ]


def save_to_csv(coef_df):
    coef_no_intercept = coef_df[coef_df['Feature'] != 'Intercept'].copy()
    top_positive = coef_no_intercept[coef_no_intercept['Coefficient'] > 0].nlargest(8, 'Coefficient')
    top_negative = coef_no_intercept[coef_no_intercept['Coefficient'] < 0].nsmallest(8, 'Coefficient')
    
    export_data = []
    
    for _, row in top_positive.iterrows():
        export_data.append({
            'word': row['Feature'],
            'coefficient_up': row['Coefficient'],
            'coefficient_down': '',
            'p-value': row['P-value']
        })
    
    for _, row in top_negative.iterrows():
        export_data.append({
            'word': row['Feature'],
            'coefficient_up': '',
            'coefficient_down': row['Coefficient'],
            'p-value': row['P-value']
        })
    
    export_df = pd.DataFrame(export_data)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, 'viral_properties.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_df.to_csv(output_path, index=False)


def plot_correlation_matrix(df, numeric_features_list, output_file=None, show=True, correlation_threshold=0.95):
    """
    Plot correlation matrix for all numeric features.
    
    Args:
        df: Dataframe with numeric features (unscaled, since correlation is scale-invariant)
        numeric_features_list: List of numeric feature names
        output_file: Optional path to save the figure
        show: Whether to display the plot
        correlation_threshold: Threshold for considering features highly correlated (default: 0.95)
    
    Returns:
        list: List of tuples (feature1, feature2, correlation_value) for highly correlated pairs
    """

    feature_data = df[numeric_features_list].select_dtypes(include=[np.number])
    corr_matrix = feature_data.corr()
    
    highly_correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > correlation_threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                highly_correlated_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
    
    fig, ax = plt.subplots(figsize=(16, 14))
    labels = corr_matrix.columns
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax,
        vmin=-1,
        vmax=1
    )
    
    ax.set_xticks(np.arange(len(corr_matrix.columns)) + 0.5)
    ax.set_yticks(np.arange(len(corr_matrix.columns)) + 0.5)
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    
    ax.set_title('Correlation Matrix of Numeric Features', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return highly_correlated_pairs


def logistic_regression(df, features_list=None):
    """
    Train a logistic regression model to predict virality (is_viral) using selected features.
    
    Args:
        df: DataFrame with is_viral column and various features
        save_to_csv: Whether to save results to CSV
        features_list: List of numeric features to use. If None, uses the global numeric_features_list.
                      The function will always check for correlations only among features in this list.
    """
    if features_list is None:
        features_list = NUMERIC_FEATURES_LIST
    
    df_model = df.copy()
    
    df_model['is_viral_numeric'] = df_model['is_viral'].astype(int)
    
    selected_features = [col for col in features_list if col in df_model.columns]
    
    print(f"Viral posts: {df_model['is_viral_numeric'].sum()} ({df_model['is_viral_numeric'].mean()*100:.2f}%)")
    print(f"Non-viral posts: {(df_model['is_viral_numeric'] == 0).sum()} ({(df_model['is_viral_numeric'] == 0).mean()*100:.2f}%)")
    
    scaler = StandardScaler()
    df_model_scaled = df_model.copy()
    for col in selected_features:
        if col in df_model_scaled.columns:
            scaled_values = scaler.fit_transform(df_model_scaled[[col]])
            scaled_values = np.clip(scaled_values, -6, 6) # clip to +/- 6 standard deviations (to prevent overflow in logistic regression)
            df_model_scaled[col] = scaled_values

    formula = "is_viral_numeric ~ "
    formula += " + ".join(selected_features)
    formula += " + C(source_supercluster)"
    
    print(f"\nFormula: {formula}\n")
    
    model = smf.logit(formula=formula, data=df_model_scaled).fit(
        maxiter=1000,
    )
    
    print("LOGISTIC REGRESSION RESULTS")
    print(model.summary())
    
    return model, df_model_scaled


if __name__ == "__main__":
    df = load_final_df(calculate_rss=True)
    df = get_viral_posts(df, virality_percentile=0.999, window_size='60D')
    model, df_model_scaled = logistic_regression(df)

    print("FEATURE COEFFICIENTS (Top 20 by absolute value)")
    
    coef_df = pd.DataFrame({
        'Feature': model.params.index,
        'Coefficient': model.params.values,
        'P-value': model.pvalues.values,
        'Abs_Coefficient': np.abs(model.params.values)
    }).sort_values('Abs_Coefficient', ascending=False)

    coef_df = coef_df[coef_df['P-value'] < 0.05]
    
    print(coef_df.head(20).to_string(index=False))
    
    if save_to_csv:
        save_to_csv(coef_df)
