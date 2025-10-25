import os
import json
import pandas as pd
import csv
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')

def count_rows_in_file():
    file_path = os.path.join(DATA_FOLDER, 'RS_2014-01', "RS_2014-01")
    cnt = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    print(f"Total number of lines: {cnt}")

def open_file_dumps():
    file_path = os.path.join(DATA_FOLDER, 'RS_2014-01', "RS_2014-01")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            filtered = {
                "id": obj.get("id"),
                "title": obj.get("title"),
                "selftext": obj.get("selftext").replace('\n', ''),
                "subreddit": obj.get("subreddit"),
                "upvote": obj.get("ups"),
                "num_comments": obj.get("num_comments"),
                "saved": obj.get("saved"),
                "over_18": obj.get("over_18"),
                "name": obj.get("name"),
                "gilded": obj.get("gilded"),
                "archived": obj.get("archived"),
                "author": obj.get("author"),
                "is_self": obj.get("is_self"),
                "url": obj.get("url"),
                #"secure_media": obj.get("secure_media"),
                "created": obj.get("created"),
                "stickied": obj.get("stickied"),
                "score": obj.get("score"),
                "edited": obj.get("edited"),
                "quarantine": obj.get("quarantine"),
                "created_utc": obj.get("created_utc"),
                #"media_embed": obj.get("media_embed"),
                "domain": obj.get("domain"),
                "hide_score": obj.get("hide_score"),
                "downs": obj.get("downs"),
                #"thumbnail": obj.get("thumbnail"),
                #"secure_media_embed": obj.get("secure_media_embed"),
                "permalink": obj.get("permalink"),
                #"media": obj.get("media"),
                "retrieved_on": obj.get("retrieved_on")
            }
            data.append(filtered)
    df = pd.DataFrame(data)
    return df

def check_if_ids_match(big_df, actual_dataset):
    ids1 = set(big_df['id'].astype(str))
    ids2 = set(actual_dataset['POST_ID'].astype(str))
    common_ids = ids1.intersection(ids2)
    return common_ids

def make_enhanced_df(common_ids, big_df, actual_dataset):
    filtered_big_df = big_df[big_df['id'].astype(str).isin(common_ids)].copy()
    filtered_actual_dataset = actual_dataset[actual_dataset['POST_ID'].astype(str).isin(common_ids)].copy()

    filtered_big_df.set_index(filtered_big_df['id'].astype(str), inplace=True)
    filtered_actual_dataset.set_index(filtered_actual_dataset['POST_ID'].astype(str), inplace=True)

    enhanced_df = filtered_actual_dataset.join(filtered_big_df[['title', 'selftext', 'subreddit']], how='left', lsuffix='_actual', rsuffix='_big')

    with open(os.path.join(os.path.dirname(__file__), '..', '..','..','data', 'enhanced_dataset.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(enhanced_df.columns)
        for row in enhanced_df.itertuples(index=False):
            writer.writerow(row)
    return enhanced_df

def plot_post_histograms(df):
    """
    Reads the enhanced CSV file and plots histograms of title length, body length, and upvotes.
    """
    title_lengths = df['title'].fillna('').apply(len)
    body_lengths = df['selftext'].fillna('').apply(len)
    upvotes = df['upvote'].fillna(0)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(title_lengths, bins=30, color='skyblue')
    axs[0].set_title('Title Lengths')
    axs[0].set_xlabel('Characters')
    axs[0].set_ylabel('Count')

    axs[1].hist(body_lengths, bins=30, color='salmon')
    axs[1].set_title('Body Lengths')
    axs[1].set_xlabel('Characters')
    axs[1].set_ylabel('Count')

    axs[2].hist(upvotes, bins=30, color='lightgreen')
    axs[2].set_title('Upvotes')
    axs[2].set_xlabel('Upvotes')
    axs[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


def plot_graphs_from_enhanced_csv():
    """
    Reads the enhanced CSV file and reddit_df_merged.pkl, compares IDs, and plots the common posts count and post histograms.
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..','..','data', 'enhanced_dataset.csv')
    pickle_path = os.path.join(os.path.dirname(__file__), '..', '..','..','data', 'reddit_df_merged.pkl')
    enhanced_df = pd.read_csv(csv_path)
    print(f"Loaded {len(enhanced_df)} rows from enhanced_dataset.csv")
    reddit_df = load_reddit_df(pickle_path)
    print(f"Loaded {len(reddit_df)} rows from reddit_df_merged.pkl")
    plot_common_posts_count(enhanced_df, reddit_df)
    plot_post_histograms(enhanced_df)

def plot_common_posts_count(enhanced_df, reddit_df):
    """
    Plots a bar graph showing the number of posts from enhanced_df found in reddit_df (common ids).
    Filters reddit_df for posts from January 2014.
    """
    reddit_df_jan2014 = reddit_df[reddit_df['TIMESTAMP'].astype(str).str.startswith('2014-01')]

    ids1 = set(enhanced_df['POST_ID'].astype(str))
    ids2 = set(reddit_df_jan2014['POST_ID'].astype(str))
    common_ids = ids1.intersection(ids2)
    n_common = len(common_ids)
    n_enhanced = len(enhanced_df)
    n_jan2014 = len(reddit_df_jan2014)
    plt.figure(figsize=(6,4))
    plt.bar(['Common IDs', 'Enhanced CSV', 'Reddit Jan 2014'], [n_common, n_enhanced, n_jan2014], color=['green', 'blue', 'orange'])
    plt.ylabel('Number of Posts')
    plt.title('Number of Common Posts (Jan 2014)')
    plt.show()



if __name__ == "__main__":
    big_df = open_file_dumps()
    print("Created big df from file dumps.")
    actual_dataset = load_reddit_df(os.path.join(DATA_FOLDER, 'reddit_df_merged.pkl'))
    print("Loaded actual dataset.")
    common_ids = check_if_ids_match(big_df, actual_dataset)
    print(f"Number of common IDs: {len(common_ids)}")
    make_enhanced_df(common_ids, big_df, actual_dataset)

    #count_rows_in_file()

