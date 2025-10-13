import os
import json
import pandas as pd
import csv

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
        for line in f:
            obj = json.loads(line)
            filtered = {
                "id": obj.get("id"),
                "title": obj.get("title"),
                "selftext": obj.get("selftext"),
                "subreddit": obj.get("subreddit"),
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

    with open(os.path.join(os.path.dirname(__file__), 'check.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(enhanced_df.columns)
        for row in enhanced_df.itertuples(index=False):
            writer.writerow(row)
    return enhanced_df

if __name__ == "__main__":
    big_df = open_file_dumps()
    print("Created big df from file dumps.")
    actual_dataset = load_reddit_df(os.path.join(DATA_FOLDER, 'reddit_df_merged.pkl'))
    print("Loaded actual dataset.")
    common_ids = check_if_ids_match(big_df, actual_dataset)
    print(f"Number of common IDs: {len(common_ids)}")
    make_enhanced_df(common_ids, big_df, actual_dataset)

    #count_rows_in_file()
