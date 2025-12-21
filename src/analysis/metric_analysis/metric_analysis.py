from src.utils.data_utils import load_final_df
from src.analysis.virality_analysis.get_viral_posts import get_viral_posts

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd



def plot_rss_distr(df=None):
    if df is None:
        df = load_final_df()
        df = get_viral_posts(df, virality_percentile=0.98, window_size='60D')

    df = df.copy()

    x = df["virality_rss"].values
    x = x[x > 0.001]

    bins = np.logspace(
        np.log10(x.min()),
        np.log10(x.max()),
        51
    )
    plt.figure(figsize=(10,6))

    plt.hist(
        x,
        bins=bins,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.4,
    )
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("virality_rss score (log scale)")
    plt.ylabel("Count (log scale)")
    plt.title("Distribution of virality_rss score")

    plt.tight_layout()
    plt.savefig("plot.svg", format="svg")
    plt.show()

def plot_same_diff_supercluster(df=None):
    if df is None:
        df = load_final_df()
        df = get_viral_posts(df, virality_percentile=0.98, window_size='60D')

    df = df.copy()

    out = (
    df
    .groupby([df["source_supercluster"] == df["target_supercluster"], "is_viral"])
    .size()
    .unstack(fill_value=0)
    .pipe(lambda d: d.div(d.sum(axis=1), axis=0))
    )
    out.index = out.index.map({
        True: "Same Supercluster",
        False: "Different Supercluster"
    })
    out.drop(columns=[False], inplace=True)

    # Used to create CSV for Flourish
    out.to_csv("share_viral_posts_by_target.csv")

def unused_by_supercluster(df=None):
    if df is None:
        df = load_final_df()

    clean_df = df.copy()
    clean_df = clean_df.iloc[:, list(range(92))]
    clean_df = clean_df.drop(columns=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID"])
    clean_df["upvote_ratio"] = df["upvote_ratio"]
    clean_df["subreddit_subscribers"] = df["subreddit_subscribers"]
    clean_df["ups"] = df["ups"]
    clean_df["downs"] = df["downs"]
    clean_df["source_supercluster"] = df["source_supercluster"]
    clean_df["target_supercluster"] = df["target_supercluster"]

    nan_df = clean_df[clean_df["ups"].isna()]
    non_nan_df = clean_df[~clean_df["ups"].isna()]

    mask = non_nan_df['upvote_ratio'] > 0
    total_votes = pd.Series(0.0, index=non_nan_df.index) # assumption: if ups = 0 or upvote_ratio = 0 we assume downs = 0
    downs = pd.Series(0.0, index=non_nan_df.index)

    total_votes.loc[mask] = non_nan_df.loc[mask, 'ups'] / non_nan_df.loc[mask, 'upvote_ratio']
    downs.loc[mask] = total_votes.loc[mask] - non_nan_df.loc[mask, 'ups']

    subreddit_subscribers = np.maximum(non_nan_df["subreddit_subscribers"], 0)

    non_nan_df = non_nan_df.assign(
        total_votes=total_votes,
        downs=downs,
        subreddit_subscribers=subreddit_subscribers,
        virality_rss=((2*non_nan_df["ups"] + non_nan_df["downs"]) / np.sqrt(subreddit_subscribers + 1))
    )

    non_nan_df = get_viral_posts(non_nan_df, virality_percentile=0.98, window_size='60D')

    nan_df['sentiment_category'] = nan_df['Sentiment_compound_vader'].apply(
        lambda x: 'Negative' if x < -0.05 else ('Neutral' if x <= 0.05 else 'Positive')
    )

    counts = nan_df.groupby(['source_supercluster', 'sentiment_category']).size().reset_index(name='count')

    # CSV for flourish graph
    counts.to_csv('sentiment_counts_by_supercluster_unscraped.csv', index=False)

    counts_rel = (
        clean_df
        .assign(ups_is_na=clean_df['ups'].isna())
        .groupby(['source_supercluster', 'ups_is_na'])
        .size()
        .unstack(fill_value=0)
    )
    counts_rel = counts_rel.div(counts_rel.sum(axis=0), axis=1)
    counts_rel = (
        counts_rel
        .rename(columns={False: 'ups_present', True: 'ups_missing'})
        .reset_index()
    )

    # CSV for flourish
    counts_rel.to_csv('unscraped_counts_by_supercluster.csv', index=False)

def get_random_forest_counts(df=None):
    if df is None:
        df = load_final_df()

    clean_df = df.copy()
    clean_df = clean_df.iloc[:, list(range(92))]
    clean_df = clean_df.drop(columns=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID"])
    clean_df["upvote_ratio"] = df["upvote_ratio"]
    clean_df["subreddit_subscribers"] = df["subreddit_subscribers"]
    clean_df["ups"] = df["ups"]
    clean_df["downs"] = df["downs"]
    clean_df["source_supercluster"] = df["source_supercluster"]
    clean_df["target_supercluster"] = df["target_supercluster"]

    nan_df = clean_df[clean_df["ups"].isna()]
    non_nan_df = clean_df[~clean_df["ups"].isna()]

    mask = non_nan_df['upvote_ratio'] > 0
    total_votes = pd.Series(0.0, index=non_nan_df.index) # assumption: if ups = 0 or upvote_ratio = 0 we assume downs = 0
    downs = pd.Series(0.0, index=non_nan_df.index)

    total_votes.loc[mask] = non_nan_df.loc[mask, 'ups'] / non_nan_df.loc[mask, 'upvote_ratio']
    downs.loc[mask] = total_votes.loc[mask] - non_nan_df.loc[mask, 'ups']

    subreddit_subscribers = np.maximum(non_nan_df["subreddit_subscribers"], 0)

    non_nan_df = non_nan_df.assign(
        total_votes=total_votes,
        downs=downs,
        subreddit_subscribers=subreddit_subscribers,
        virality_rss=((2*non_nan_df["ups"] + non_nan_df["downs"]) / np.sqrt(subreddit_subscribers + 1))
    )

    non_nan_df = get_viral_posts(non_nan_df, virality_percentile=0.98, window_size='60D')

    non_nan_df = pd.get_dummies(non_nan_df, columns=["source_supercluster", "target_supercluster", "source_type"], drop_first=True)
    nan_df = pd.get_dummies(nan_df, columns=["source_supercluster", "target_supercluster", "source_type"], drop_first=True)
    nan_df = nan_df.reindex(columns=non_nan_df.columns, fill_value=0)
    nan_df = nan_df.drop(columns=["is_viral", "z_score", "moving_std_virality", "moving_avg_virality","virality_rss",
                                "upvote_ratio", "subreddit_subscribers", "ups", "downs", "total_votes", "TIMESTAMP"])
    non_nan_df = non_nan_df.drop(columns=["TIMESTAMP", "z_score", "moving_std_virality", "moving_avg_virality","virality_rss",
                              "upvote_ratio", "subreddit_subscribers", "ups", "downs", "total_votes"])

    X = non_nan_df.drop(columns=["is_viral"])
    y = non_nan_df["is_viral"] 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
        class_weight={0:1, 1:70}
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    nan_posts_pred = rf.predict(nan_df)
    nan_df = nan_df.assign(is_viral_pred=nan_posts_pred)

    mask = nan_df.filter(like="source_supercluster_").sum(axis=1) > 0

    nan_df["source_supercluster"] = (
        nan_df.filter(like="source_supercluster_")
            .idxmax(axis=1)
            .where(mask)
            .str.replace("source_supercluster_", "")
    )

    nan_df.loc[~mask, "source_supercluster"] = "Gaming & Interactive Entertainment"

    fractions = nan_df.groupby("source_supercluster")["is_viral_pred"].mean().sort_values(ascending=False).reset_index()
    fractions["is_viral_pred"] = fractions["is_viral_pred"]*100
    fractions["Original"] = 2

    # CSV for flourish
    fractions.to_csv("predicted_viral_fraction_by_supercluster_unscraped.csv")

    importances = rf.feature_importances_
    feature_names = X_train.columns

    # Sort and select top 10
    idx = np.argsort(importances)[-10:]
    top_features = feature_names[idx]
    top_importances = importances[idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(top_features[::-1], top_importances[::-1])
    plt.ylabel("Feature Importance")
    plt.xticks(rotation=90)
    plt.title("Top 10 Feature Importances")
    plt.savefig("top_10_feature_importances.svg", format="svg", bbox_inches='tight')
    plt.show()

def supercluster_interactions(df=None):
    if df is None:
        df = load_final_df()

    df = df.copy()
    interactions = df.groupby(["source_supercluster", "target_supercluster"]).size().reset_index()
    interactions["source_supercluster"] = interactions["source_supercluster"].str.split('&').str[0].str.strip()
    interactions["target_supercluster"] = interactions["target_supercluster"].str.split('&').str[0].str.strip()
    interactions.to_csv("supercluster_interactions.csv", index=False)

def share_of_posts_used(df=None):
    if df is None:
        df = load_final_df()

    d = {}
    df_met = df.copy()
    original_size = df_met.shape[0]
    df_met = df_met[df_met["source_supercluster"].notna()]
    not_in_embeddings = original_size-df_met.shape[0]
    df_met = df_met[df_met["ups"].notna()]
    not_in_scraped = original_size - not_in_embeddings-df_met.shape[0]
    unique_ids = df_met.POST_ID.nunique()
    double_ids = original_size - not_in_embeddings - not_in_scraped - unique_ids

    d = {}
    d["Final Set"] = unique_ids
    d["Double IDs"] = double_ids
    d["Not Scraped"] = not_in_scraped
    d["Not in Embeddings"] = not_in_embeddings

    # CSV for flourish
    pd.Series(d).to_csv('share_of_posts_used.csv', header=['Value'])