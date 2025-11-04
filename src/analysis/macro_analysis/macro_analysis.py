import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')

def prepare_data(path):
    """
    Load Reddit dataframe from pickle file.
    
    Args:
        path: Path to the pickle file
    
    Returns:
        pandas.DataFrame: Loaded Reddit dataframe
    """
    return load_reddit_df(path)

def transform_link_sentiment(df):
    """
    Transform LINK_SENTIMENT column by replacing -1 with 0.
    
    Args:
        df: DataFrame with LINK_SENTIMENT column
    
    Returns:
        pandas.DataFrame: DataFrame with transformed LINK_SENTIMENT values
    """
    df = df.copy()
    if 'LINK_SENTIMENT' in df.columns:
        df['LINK_SENTIMENT'] = df['LINK_SENTIMENT'].replace(-1, 0)
    return df

def logistic_regression_all(df, title):
    """
    Perform logistic regression using all 86 property features to predict LINK_SENTIMENT.
    
    Args:
        df: DataFrame with property features and LINK_SENTIMENT
        title: Title suffix for output file
    
    Returns:
        statsmodels.discrete.discrete_model.LogitResults: Fitted logistic regression model results
    """
    
    properties_names = [
        "Num_characters", "Num_characters_no_space", "Frac_alpha", "Frac_digits", "Frac_uppercase", "Frac_whitespace", "Frac_special", "Num_words", "Num_unique_words", "Num_long_words", "Avg_word_length", "Num_unique_stopwords", "Frac_stopwords", "Num_sentences", "Num_long_sentences", "Avg_chars_per_sentence", "Avg_words_per_sentence", "Automated_readability_index", "Sentiment_pos_vader", "Sentiment_neg_vader", "Sentiment_compound_vader", "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"
    ]
    X = df[properties_names].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['LINK_SENTIMENT'].astype(int)
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    # Format and write results to file
    summary_str = result.summary().as_text()
    with open(os.path.join(os.path.dirname(__file__), 'graphs', f"logit_results_all_{title}.txt"), "w", encoding="utf-8") as f:
        f.write(summary_str)
    return result

def logistic_regression_subset(df, title):
    """
    Perform logistic regression using a subset of key features to predict LINK_SENTIMENT.
    
    Args:
        df: DataFrame with property features and LINK_SENTIMENT
        title: Title suffix for output file
    
    Returns:
        statsmodels.discrete.discrete_model.LogitResults: Fitted logistic regression model results
    """
    subset = ["Num_characters", "Num_words", "Avg_word_length", "Sentiment_compound_vader", "LIWC_Affect"]
    X = df[subset].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['LINK_SENTIMENT'].astype(int)
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    summary_str = result.summary().as_text()
    with open(os.path.join(os.path.dirname(__file__), 'graphs', f"logit_results_subset_{title}.txt"), "w", encoding="utf-8") as f:
        f.write(summary_str)
    return result

def logistic_regression_text_features(df, title):
    """
    Perform logistic regression using only text-based features to predict LINK_SENTIMENT.
    
    Args:
        df: DataFrame with property features and LINK_SENTIMENT
        title: Title suffix for output file
    
    Returns:
        statsmodels.discrete.discrete_model.LogitResults: Fitted logistic regression model results
    """
    features = ["Num_words", "Frac_uppercase", "Avg_words_per_sentence", "Automated_readability_index"]
    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['LINK_SENTIMENT'].astype(int)
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    summary_str = result.summary().as_text()
    with open(os.path.join(os.path.dirname(__file__), 'graphs', f"logit_results_text_features_{title}.txt"), "w", encoding="utf-8") as f:
        f.write(summary_str)
    return result

def scatterplot_link_sentiment_vs_compound(df, title):
    """
    Create scatterplot comparing LINK_SENTIMENT vs VADER compound sentiment.
    
    Args:
        df: DataFrame with LINK_SENTIMENT and Sentiment_compound_vader columns
        title: Title suffix for output file
    """
    plt.figure(figsize=(8, 6))
    x = pd.to_numeric(df['Sentiment_compound_vader'], errors='coerce')
    y = pd.to_numeric(df['LINK_SENTIMENT'], errors='coerce')
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Compound Sentiment (VADER)')
    plt.ylabel('LINK_SENTIMENT')
    plt.title('Scatterplot: LINK_SENTIMENT vs Compound Sentiment (VADER)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'graphs', f'scatter_link_sentiment_vs_compound_{title}.png'))
    plt.close()


def boxplot_sentiment_vs_compound(df, title):
    """
    Create boxplot showing VADER compound sentiment distribution by LINK_SENTIMENT.
    
    Args:
        df: DataFrame with LINK_SENTIMENT and Sentiment_compound_vader columns
        title: Title suffix for output file
    """
    plt.figure(figsize=(8, 6))
    df_plot = df.copy()
    df_plot['Sentiment_compound_vader'] = pd.to_numeric(df_plot['Sentiment_compound_vader'], errors='coerce')
    df_plot['LINK_SENTIMENT'] = pd.to_numeric(df_plot['LINK_SENTIMENT'], errors='coerce')
    sns.boxplot(x='LINK_SENTIMENT', y='Sentiment_compound_vader', data=df_plot)
    plt.xlabel('LINK_SENTIMENT')
    plt.ylabel('Compound Sentiment (VADER)')
    plt.title('Boxplot: Compound Sentiment by LINK_SENTIMENT')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'graphs', f'boxplot_sentiment_vs_compound_{title}.png'))
    plt.close()

def boxplot_pos_sentiment_vs_link(df, title):
    """
    Create boxplot showing VADER positive sentiment distribution by LINK_SENTIMENT.
    
    Args:
        df: DataFrame with LINK_SENTIMENT and Sentiment_pos_vader columns
        title: Title suffix for output file
    """
    plt.figure(figsize=(8, 6))
    df_plot = df.copy()
    df_plot['Sentiment_pos_vader'] = pd.to_numeric(df_plot['Sentiment_pos_vader'], errors='coerce')
    df_plot['LINK_SENTIMENT'] = pd.to_numeric(df_plot['LINK_SENTIMENT'], errors='coerce')
    sns.boxplot(x='LINK_SENTIMENT', y='Sentiment_pos_vader', data=df_plot)
    plt.xlabel('LINK_SENTIMENT')
    plt.ylabel('Positive Sentiment (VADER)')
    plt.title('Boxplot: Positive Sentiment by LINK_SENTIMENT')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'graphs', f'boxplot_pos_sentiment_vs_link_{title}.png'))
    plt.close()

def boxplot_neg_sentiment_vs_link(df, title):
    """
    Create boxplot showing VADER negative sentiment distribution by LINK_SENTIMENT.
    
    Args:
        df: DataFrame with LINK_SENTIMENT and Sentiment_neg_vader columns
        title: Title suffix for output file
    """
    plt.figure(figsize=(8, 6))
    df_plot = df.copy()
    df_plot['Sentiment_neg_vader'] = pd.to_numeric(df_plot['Sentiment_neg_vader'], errors='coerce')
    df_plot['LINK_SENTIMENT'] = pd.to_numeric(df_plot['LINK_SENTIMENT'], errors='coerce')
    sns.boxplot(x='LINK_SENTIMENT', y='Sentiment_neg_vader', data=df_plot)
    plt.xlabel('LINK_SENTIMENT')
    plt.ylabel('Negative Sentiment (VADER)')
    plt.title('Boxplot: Negative Sentiment by LINK_SENTIMENT')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'graphs', f'boxplot_neg_sentiment_vs_link_{title}.png'))
    plt.close()

if __name__ == "__main__":
    # get_post_data()
    df_body = prepare_data(os.path.join(DATA_FOLDER, 'reddit_df_body.pkl'))
    scatterplot_link_sentiment_vs_compound(df_body, "body")
    boxplot_sentiment_vs_compound(df_body, "body")
    boxplot_pos_sentiment_vs_link(df_body, "body")
    boxplot_neg_sentiment_vs_link(df_body, "body")
    df_body = transform_link_sentiment(df_body)
    logistic_regression_all(df_body, "body")
    logistic_regression_subset(df_body, "body")
    logistic_regression_text_features(df_body, "body")

    df_title = prepare_data(os.path.join(DATA_FOLDER, 'reddit_df_title.pkl'))
    scatterplot_link_sentiment_vs_compound(df_title, "title")
    boxplot_sentiment_vs_compound(df_title, "title")
    boxplot_pos_sentiment_vs_link(df_title, "title")
    boxplot_neg_sentiment_vs_link(df_title, "title")
    df_title = transform_link_sentiment(df_title)
    logistic_regression_all(df_title, "title")
    logistic_regression_subset(df_title, "title")
    logistic_regression_text_features(df_title, "title")