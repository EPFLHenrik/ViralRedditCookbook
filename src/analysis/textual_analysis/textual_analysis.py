import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm
from src.utils.data_utils import load_final_df
from src.data.preprocess_text import preporcess_text
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "figures", "text_analysis")


def preprocess_text_spacy(texts, nlp, chunk_size=1000):
    """
    Preprocess texts using spaCy: tokenization, stopword removal, and lemmatization.
    
    Args:
        texts: Series or list of text strings
        nlp: Loaded spaCy language model
        chunk_size: Number of texts to process at once (default: 1000)
    
    Returns:
        List of preprocessed text strings (space-separated lemmatized tokens)
    """
    processed_texts = []
    
    texts_list = [str(text) if not pd.isna(text) and text != "" else "" for text in texts]
    total_texts = len(texts_list)
    
    batch_size = 1000
    
    for i in tqdm(range(0, total_texts, chunk_size), desc="Processing texts", total=total_texts):
        chunk = texts_list[i:i + chunk_size]
        
        docs = nlp.pipe(chunk, batch_size=batch_size, n_process=1)
        
        for doc in docs:
            if doc.text == "":
                processed_texts.append("")
            else:
                tokens = [
                    token.lemma_.lower() 
                    for token in doc 
                    if not token.is_stop 
                    and not token.is_punct 
                    and not token.is_space
                    and len(token.lemma_) > 1  # Filter out single character tokens
                ]
                processed_texts.append(" ".join(tokens))
    
    return processed_texts


def preprocess_text_simple(texts):
    """
    Simple preprocessing: split by spaces, remove punctuation, and remove stopwords.
    
    Args:
        texts: Series or list of text strings
    
    Returns:
        List of preprocessed text strings (space-separated words, no punctuation, no stopwords)
    """
    processed_texts = []
    
    translator = str.maketrans('', '', string.punctuation)
    stopwords = STOP_WORDS
    
    for text in tqdm(texts, desc="Processing texts (simple)"):
        if pd.isna(text) or text == "":
            processed_texts.append("")
            continue
        
        text_clean = str(text).translate(translator)
        words = [
            word.lower() 
            for word in text_clean.split() 
            if word.strip() and word.lower() not in stopwords
        ]
        processed_texts.append(" ".join(words))
    
    return processed_texts


def get_viral_words(df, n_top_words=50, alpha=1.0, min_df=5, max_df=0.95, random_state=42, use_simple_preprocessing=False, text_column='title'):
    """
    Analyze which words in post text most influence upvote counts using Lasso regression.
    
    Optimal lambda (alpha) is found using 5-fold cross-validation, then the final model
    is fit on all data with the optimal alpha.
    
    Args:
        df: DataFrame with text column and 'virality_rss' columns
        n_top_words: Number of top words to return (default: 50)
        alpha: Lasso regularization strength (ignored - optimal alpha is found via CV) (default: 1.0)
        min_df: Minimum document frequency for words (default: 5)
        max_df: Maximum document frequency for words (default: 0.95)
        random_state: Random state for reproducibility (default: 42)
        verbose: If True, show progress bars (default: True)
        use_simple_preprocessing: If True, use simple preprocessing (split + remove punctuation) instead of spaCy (default: False)
        text_column: Name of the text column to analyze (default: 'title')
    
    Returns:
        Dictionary with keys 'positive' and 'negative', each containing a DataFrame with
        n_top_words words. Each DataFrame has columns: 'word', 'coefficient', 'abs_coefficient'
    """
    df_clean = df.dropna(subset=[text_column, 'virality_rss']).copy()
    
    df_clean = df_clean[df_clean[text_column].str.strip() != ''].copy()
    
    df_clean['log_ups'] = np.log1p(df_clean['virality_rss'])
    
    print(f"Processing {len(df_clean)} posts with valid text and upvote data...")
    
    if use_simple_preprocessing:
        print(f"Preprocessing {text_column} with simple method (split by spaces, remove punctuation)...")
        processed_texts = preprocess_text_simple(df_clean[text_column])
    else:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            raise OSError(
                "Install spacy english model with: python -m spacy download en_core_web_sm"
            )
        
        print(f"Preprocessing {text_column} with spaCy")

        processed_texts = preprocess_text_spacy(df_clean[text_column], nlp)
    
    valid_mask = [len(text.strip()) > 0 for text in processed_texts]
    df_clean = df_clean[valid_mask].copy()
    processed_texts = [text for text, valid in zip(processed_texts, valid_mask) if valid]
    
    print(f"After preprocessing: {len(df_clean)} valid posts")
    
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=20000,
        ngram_range=(1, 1),
        lowercase=False
    )
    
    X = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    y = df_clean['log_ups'].values
    
    print(f"Created {X.shape[1]} features from {X.shape[0]} posts")
    
    print(f"Finding optimal lambda (alpha) using 5-fold cross-validation...")
    alpha_values = np.logspace(-4, 1, 20)
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    cv_scores = []
    for alpha in tqdm(alpha_values, desc="Testing alpha values"):
        lasso_temp = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
        scores = cross_val_score(lasso_temp, X, y, cv=kfold, scoring='r2', n_jobs=-1)
        cv_scores.append(scores.mean())
    
    best_idx = np.argmax(cv_scores)
    optimal_alpha = alpha_values[best_idx]
    best_cv_score = cv_scores[best_idx]
    print(f"Optimal alpha: {optimal_alpha:.6f} (5-fold CV R^2: {best_cv_score})")
    
    print(f"Fitting final Lasso regression with alpha={optimal_alpha:.6f}...")
    lasso = Lasso(alpha=optimal_alpha, random_state=random_state, max_iter=2000)
    lasso.fit(X, y)
    
    coefficients = lasso.coef_
    
    results_df = pd.DataFrame({
        'word': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    results_df = results_df.sort_values('abs_coefficient', ascending=False)
    
    results_df = results_df[results_df['abs_coefficient'] > 1e-10]
    
    r2_score = lasso.score(X, y)
    print(f"\nModel R^2 score: {r2_score}")
    print(f"Number of non-zero coefficients: {(np.abs(coefficients) > 1e-10).sum()}")
    
    positive_df = results_df[results_df['coefficient'] > 0].sort_values('coefficient', ascending=False).head(n_top_words)
    negative_df = results_df[results_df['coefficient'] < 0].sort_values('coefficient', ascending=True).head(n_top_words)
    
    return {
        'positive': positive_df,
        'negative': negative_df
    }


def save_to_csv(positive_words, negative_words, cluster_dir):
    export_data = []
    for _, row in positive_words.iterrows():
        export_data.append({
            'word': row['word'],
            'coefficient_up': row['coefficient'],
            'coefficient_down': ''
        })
    
    for _, row in negative_words.iterrows():
        export_data.append({
            'word': row['word'],
            'coefficient_up': '',
            'coefficient_down': row['coefficient']
        })
    
    export_df = pd.DataFrame(export_data)
    csv_path = os.path.join(cluster_dir, 'viral_words_title.csv')
    export_df.to_csv(csv_path, index=False)
    
    
if __name__ == "__main__":
    df = load_final_df()
    df = preporcess_text(df)
    print(f"Total posts: {len(df)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    superclusters = df['source_supercluster'].unique()
    print(f"Found {len(superclusters)} unique source superclusters")
    
    for cluster in superclusters:
        print(f"\nCLUSTER: {cluster}")

        cluster_df = df[df['source_supercluster'] == cluster].copy()
        cluster_dir = os.path.join(OUTPUT_DIR, str(cluster))
        os.makedirs(cluster_dir, exist_ok=True)
        
        viral_words_cluster = get_viral_words(
            cluster_df, 
            n_top_words=8, 
            alpha=0.001, 
            use_simple_preprocessing=True, 
            text_column='title',
        )
        
        positive_words = viral_words_cluster['positive']
        negative_words = viral_words_cluster['negative']
        
        save_to_csv(positive_words, negative_words, cluster_dir)

        print(f"\nTop {len(positive_words)} positive words ({cluster}):")
        print(positive_words[['word', 'coefficient']].to_string(index=False))
        
        print(f"\nTop {len(negative_words)} negative words ({cluster}):")
        print(negative_words[['word', 'coefficient']].to_string(index=False))
