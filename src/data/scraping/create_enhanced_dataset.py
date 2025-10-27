import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')


def load_file(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"POST_ID": "string"})
    return df

def fix_csv(df):
    df['POST_ID'] = df['POST_ID'].astype(str).str[:6]
    df = df.dropna(how='all')
    df = df.drop_duplicates().reset_index(drop=True)
    df['POST_ID'].replace('', pd.NA, inplace=True)
    df = df.dropna(subset=['POST_ID']).reset_index(drop=True)
    return df

def merge_csvs(df1, df2):
    result = pd.concat([df1, df2], ignore_index=True)
    return result

def save_enhanced_df(enhanced_df):
    enhanced_df.to_pickle(os.path.join(DATA_FOLDER, "enhanced_reddit_df.pkl"))
    enhanced_df.to_csv(os.path.join(DATA_FOLDER, "enhanced_reddit_df.csv"))


if __name__ == "__main__":
    df1 = load_file(os.path.join(DATA_FOLDER, 'reddit_enhanced_backup_2.csv'))
    df2 = load_file(os.path.join(DATA_FOLDER, 'reddit_enhanced_backup_3.csv'))
    df1 = fix_csv(df1)
    df2 = fix_csv(df2)
    enhanced_df = merge_csvs(df1, df2)
    save_enhanced_df(enhanced_df)