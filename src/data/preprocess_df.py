import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def preprocess_dataframe(
    load_type="body",
    body_tsv_file="soc-redditHyperlinks-body.tsv",
    title_tsv_file="soc-redditHyperlinks-title.tsv",
    force_reload=False,
    ignore_clustering=True,
):
    """
    Preprocess Reddit hyperlink data into a pandas DataFrame with expanded property columns.
    
    Args:
        load_type: Type of data to load. Options: 'body' (body hyperlinks only), 'title' (title hyperlinks only), 'merged' (both)
        body_tsv_file: Filename for body hyperlinks TSV file (will be read from data folder)
        title_tsv_file: Filename for title hyperlinks TSV file (will be read from data folder)
        force_reload: If True, force reload from TSV even if pickle exists
        ignore_clustering: If True, attach cluster names to the dataframe
    
    Returns:
        pandas.DataFrame: Preprocessed dataframe with properties expanded into individual columns
    """
    data_folder = DATA_DIR

    if load_type == "body":
        tsv_file = os.path.join(data_folder, body_tsv_file)
        pickle_file = os.path.join(data_folder, "reddit_df_body.pkl")
    elif load_type == "title":
        tsv_file = os.path.join(data_folder, title_tsv_file)
        pickle_file = os.path.join(data_folder, "reddit_df_title.pkl")
    elif load_type == "merged":
        tsv_file = [
            os.path.join(data_folder, body_tsv_file),
            os.path.join(data_folder, title_tsv_file),
        ]
        pickle_file = os.path.join(data_folder, "reddit_df_merged.pkl")
    else:
        raise ValueError(
            f"load_type must be 'body', 'title', or 'merged' (got {load_type!r})"
        )

    if not force_reload and os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            df = pd.read_pickle(f)
            return df

    if load_type == "merged":
        body_df = pd.read_csv(tsv_file[0], sep="\t")
        title_df = pd.read_csv(tsv_file[1], sep="\t")
        body_df["source_type"] = "body"
        title_df["source_type"] = "title"
        df = pd.concat([body_df, title_df], ignore_index=True)
    else:
        df = pd.read_csv(tsv_file, sep="\t")

    properties_names = [
        "Num_characters",
        "Num_characters_no_space",
        "Frac_alpha",
        "Frac_digits",
        "Frac_uppercase",
        "Frac_whitespace",
        "Frac_special",
        "Num_words",
        "Num_unique_words",
        "Num_long_words",
        "Avg_word_length",
        "Num_unique_stopwords",
        "Frac_stopwords",
        "Num_sentences",
        "Num_long_sentences",
        "Avg_chars_per_sentence",
        "Avg_words_per_sentence",
        "Automated_readability_index",
        "Sentiment_pos_vader",
        "Sentiment_neg_vader",
        "Sentiment_compound_vader",
        "LIWC_Funct",
        "LIWC_Pronoun",
        "LIWC_Ppron",
        "LIWC_I",
        "LIWC_We",
        "LIWC_You",
        "LIWC_SheHe",
        "LIWC_They",
        "LIWC_Ipron",
        "LIWC_Article",
        "LIWC_Verbs",
        "LIWC_AuxVb",
        "LIWC_Past",
        "LIWC_Present",
        "LIWC_Future",
        "LIWC_Adverbs",
        "LIWC_Prep",
        "LIWC_Conj",
        "LIWC_Negate",
        "LIWC_Quant",
        "LIWC_Numbers",
        "LIWC_Swear",
        "LIWC_Social",
        "LIWC_Family",
        "LIWC_Friends",
        "LIWC_Humans",
        "LIWC_Affect",
        "LIWC_Posemo",
        "LIWC_Negemo",
        "LIWC_Anx",
        "LIWC_Anger",
        "LIWC_Sad",
        "LIWC_CogMech",
        "LIWC_Insight",
        "LIWC_Cause",
        "LIWC_Discrep",
        "LIWC_Tentat",
        "LIWC_Certain",
        "LIWC_Inhib",
        "LIWC_Incl",
        "LIWC_Excl",
        "LIWC_Percept",
        "LIWC_See",
        "LIWC_Hear",
        "LIWC_Feel",
        "LIWC_Bio",
        "LIWC_Body",
        "LIWC_Health",
        "LIWC_Sexual",
        "LIWC_Ingest",
        "LIWC_Relativ",
        "LIWC_Motion",
        "LIWC_Space",
        "LIWC_Time",
        "LIWC_Work",
        "LIWC_Achiev",
        "LIWC_Leisure",
        "LIWC_Home",
        "LIWC_Money",
        "LIWC_Relig",
        "LIWC_Death",
        "LIWC_Assent",
        "LIWC_Dissent",
        "LIWC_Nonflu",
        "LIWC_Filler",
    ]
    properties_df = df["PROPERTIES"].str.split(",", expand=True)
    properties_df.columns = properties_names
    df = pd.concat([df.drop(columns=["PROPERTIES"]), properties_df], axis=1)

    if ignore_clustering:
        df = attach_cluster_names_to_df(df)
    
    df['POST_ID'] = df['POST_ID'].astype(str).str[:6]
    df.to_pickle(pickle_file)
    return df


def attach_cluster_names_to_df(df):
    """
    Attach cluster names to dataframe based on subreddit clustering results.
    
    Args:
        df: DataFrame with SOURCE_SUBREDDIT and TARGET_SUBREDDIT columns
    
    Returns:
        pandas.DataFrame: DataFrame with added cluster columns for source and target subreddits at each level
    """
    results_dir = os.path.join(os.path.dirname(__file__), "..", "analysis", "clustering_results")
    labels_named_pkl = os.path.join(results_dir, "labels_per_level_named.pkl")
    embeddings_file = os.path.join(results_dir, "processed_embeddings.npz")

    if not os.path.exists(labels_named_pkl):
        return df

    if not os.path.exists(embeddings_file):
        return df

    with open(labels_named_pkl, "rb") as f:
        labels_per_level_named = pickle.load(f)

    data = dict(np.load(embeddings_file, allow_pickle=True))
    subreddit_names = data["subreddit_names"]

    subreddit_to_clusters = {}
    for node_idx, subreddit_name in enumerate(subreddit_names):
        subreddit_to_clusters[subreddit_name] = {}
        for level in sorted(labels_per_level_named.keys()):
            if node_idx in labels_per_level_named[level]:
                cluster_name = labels_per_level_named[level][node_idx]
                subreddit_to_clusters[subreddit_name][level] = cluster_name

    for level in sorted(labels_per_level_named.keys()):
        df[f"source_cluster_level_{level}"] = df["SOURCE_SUBREDDIT"].apply(
            lambda x: subreddit_to_clusters.get(x, {}).get(level, None)
        )
        df[f"target_cluster_level_{level}"] = df["TARGET_SUBREDDIT"].apply(
            lambda x: subreddit_to_clusters.get(x, {}).get(level, None)
        )

    return df


if __name__ == "__main__":
    df = preprocess_dataframe(
        load_type="merged",
        body_tsv_file="soc-redditHyperlinks-body.tsv",
        title_tsv_file="soc-redditHyperlinks-title.tsv",
    )
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]:,} columns.")
