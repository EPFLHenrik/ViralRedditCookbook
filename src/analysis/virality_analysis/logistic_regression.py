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


FEATURE_FULL_NAMES = {
    'Num_characters': 'Number of Characters',
    'Num_characters_no_space': 'Number of Characters (No Spaces)',
    'Frac_alpha': 'Fraction of Alphabetic Characters',
    'Frac_digits': 'Fraction of Digits',
    'Frac_uppercase': 'Fraction of Uppercase Characters',
    'Frac_whitespace': 'Fraction of Whitespace Characters',
    'Frac_special': 'Fraction of Special Characters',
    'Num_words': 'Number of Words',
    'Num_unique_words': 'Number of Unique Words',
    'Num_long_words': 'Number of Long Words',
    'Avg_word_length': 'Average Word Length',
    'Num_unique_stopwords': 'Number of Unique Stopwords',
    'Frac_stopwords': 'Fraction of Stopwords',
    'Num_sentences': 'Number of Sentences',
    'Num_long_sentences': 'Number of Long Sentences',
    'Avg_chars_per_sentence': 'Average Characters per Sentence',
    'Avg_words_per_sentence': 'Average Words per Sentence',
    'Automated_readability_index': 'Automated Readability Index',
    'Sentiment_pos_vader': 'Positive Sentiment (VADER)',
    'Sentiment_neg_vader': 'Negative Sentiment (VADER)',
    'Sentiment_compound_vader': 'Compound Sentiment (VADER)',
    'LIWC_Funct': 'LIWC: Function Words',
    'LIWC_Pronoun': 'LIWC: Pronouns',
    'LIWC_Ppron': 'LIWC: Personal Pronouns',
    'LIWC_I': 'LIWC: First Person Singular',
    'LIWC_We': 'LIWC: First Person Plural',
    'LIWC_You': 'LIWC: Second Person',
    'LIWC_SheHe': 'LIWC: Third Person Singular',
    'LIWC_They': 'LIWC: Third Person Plural',
    'LIWC_Ipron': 'LIWC: Impersonal Pronouns',
    'LIWC_Article': 'LIWC: Articles',
    'LIWC_Verbs': 'LIWC: Verbs',
    'LIWC_AuxVb': 'LIWC: Auxiliary Verbs',
    'LIWC_Past': 'LIWC: Past Tense',
    'LIWC_Present': 'LIWC: Present Tense',
    'LIWC_Future': 'LIWC: Future Tense',
    'LIWC_Adverbs': 'LIWC: Adverbs',
    'LIWC_Prep': 'LIWC: Prepositions',
    'LIWC_Conj': 'LIWC: Conjunctions',
    'LIWC_Negate': 'LIWC: Negations',
    'LIWC_Quant': 'LIWC: Quantifiers',
    'LIWC_Numbers': 'LIWC: Numbers',
    'LIWC_Swear': 'LIWC: Swear Words',
    'LIWC_Social': 'LIWC: Social Words',
    'LIWC_Family': 'LIWC: Family Words',
    'LIWC_Friends': 'LIWC: Friends Words',
    'LIWC_Humans': 'LIWC: Human References',
    'LIWC_Affect': 'LIWC: Affective Processes',
    'LIWC_Posemo': 'LIWC: Positive Emotions',
    'LIWC_Negemo': 'LIWC: Negative Emotions',
    'LIWC_Anx': 'LIWC: Anxiety',
    'LIWC_Anger': 'LIWC: Anger',
    'LIWC_Sad': 'LIWC: Sadness',
    'LIWC_CogMech': 'LIWC: Cognitive Mechanisms',
    'LIWC_Insight': 'LIWC: Insight',
    'LIWC_Cause': 'LIWC: Causation',
    'LIWC_Discrep': 'LIWC: Discrepancy',
    'LIWC_Tentat': 'LIWC: Tentative',
    'LIWC_Certain': 'LIWC: Certainty',
    'LIWC_Inhib': 'LIWC: Inhibition',
    'LIWC_Incl': 'LIWC: Inclusive',
    'LIWC_Excl': 'LIWC: Exclusive',
    'LIWC_Percept': 'LIWC: Perceptual Processes',
    'LIWC_See': 'LIWC: Seeing',
    'LIWC_Hear': 'LIWC: Hearing',
    'LIWC_Feel': 'LIWC: Feeling',
    'LIWC_Bio': 'LIWC: Biological Processes',
    'LIWC_Body': 'LIWC: Body',
    'LIWC_Health': 'LIWC: Health',
    'LIWC_Sexual': 'LIWC: Sexual',
    'LIWC_Ingest': 'LIWC: Ingestion',
    'LIWC_Relativ': 'LIWC: Relativity',
    'LIWC_Motion': 'LIWC: Motion',
    'LIWC_Space': 'LIWC: Space',
    'LIWC_Time': 'LIWC: Time',
    'LIWC_Work': 'LIWC: Work',
    'LIWC_Achiev': 'LIWC: Achievement',
    'LIWC_Leisure': 'LIWC: Leisure',
    'LIWC_Home': 'LIWC: Home',
    'LIWC_Money': 'LIWC: Money',
    'LIWC_Relig': 'LIWC: Religion',
    'LIWC_Death': 'LIWC: Death',
    'LIWC_Assent': 'LIWC: Assent',
    'LIWC_Dissent': 'LIWC: Dissent',
    'LIWC_Nonflu': 'LIWC: Nonfluencies',
    'LIWC_Filler': 'LIWC: Fillers',
    'LINK_SENTIMENT': 'Link Sentiment',
    'num_body_links': 'Number of Body Links',
    'num_title_links': 'Number of Title Links',
}


def save_to_csv(coef_df):
    coef_no_intercept = coef_df[coef_df['Feature'] != 'Intercept'].copy()
    top_positive = coef_no_intercept[coef_no_intercept['Coefficient'] > 0].nlargest(8, 'Coefficient')
    top_negative = coef_no_intercept[coef_no_intercept['Coefficient'] < 0].nsmallest(8, 'Coefficient')
    
    export_data = []
    
    for _, row in top_positive.iterrows():
        feature_name = row['Feature']
        full_name = FEATURE_FULL_NAMES.get(feature_name, feature_name)
        export_data.append({
            'word': feature_name,
            'full_name': full_name,
            'coefficient_up': row['Coefficient'],
            'coefficient_down': '',
            'p-value': row['P-value']
        })
    
    for _, row in top_negative.iterrows():
        feature_name = row['Feature']
        full_name = FEATURE_FULL_NAMES.get(feature_name, feature_name)
        export_data.append({
            'word': feature_name,
            'full_name': full_name,
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
