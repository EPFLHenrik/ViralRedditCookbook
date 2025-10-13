import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import os
from tqdm import tqdm

import requests

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')

def get_post_data(path):
    df = pd.read_csv(path, sep="\t", header=0)
    properties_names = [
        "Num_characters", "Num_characters_no_space", "Frac_alpha", "Frac_digits", "Frac_uppercase", "Frac_whitespace", "Frac_special", "Num_words", "Num_unique_words", "Num_long_words", "Avg_word_length", "Num_unique_stopwords", "Frac_stopwords", "Num_sentences", "Num_long_sentences", "Avg_chars_per_sentence", "Avg_words_per_sentence", "Automated_readability_index", "Sentiment_pos_vader", "Sentiment_neg_vader", "Sentiment_compound_vader", "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"
    ]
    if 'properties' in df.columns:
        properties_df = df['properties'].str.split(',', expand=True)
        properties_df.columns = properties_names
        df = pd.concat([df.drop(columns=['properties']), properties_df], axis=1)
    import time
    found_posts = []
    batch_size = 10
    for i, post_id in enumerate(tqdm(df['POST_ID'])):
        url = f"https://www.reddit.com/api/info.json?id=t3_{post_id}"
        response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
        data = response.json()
        children = data.get('data', {}).get('children', [])
        if children:
            found_posts.append([post_id, children[0]['data']['selftext']])
        time.sleep(1)
        
        if (i + 1) % batch_size == 0 and found_posts:
            found_df = pd.DataFrame(found_posts, columns=["POST_ID", "POST_TEXT"])
            found_df.to_csv("found_posts.csv", mode='a', header=not os.path.exists("found_posts.csv"), index=False)
            found_posts = []
            
    if found_posts:
        found_df = pd.DataFrame(found_posts, columns=["POST_ID", "POST_TEXT"])
        found_df.to_csv("found_posts.csv", mode='a', header=not os.path.exists("found_posts.csv"), index=False)

def prepare_data(path):
    return load_reddit_df(path)

def transform_link_sentiment(df):
    df = df.copy()
    if 'LINK_SENTIMENT' in df.columns:
        df['LINK_SENTIMENT'] = df['LINK_SENTIMENT'].replace(-1, 0)
    return df

def logistic_regression_all(df, title):
    
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