import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')


def load_file(path):
    """
    Load CSV file with UTF-8 encoding and string POST_ID type.
    
    Args:
        path: Path to the CSV file to load
    
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"POST_ID": "string"})
    return df

def fix_csv(df):
    """
    Clean and fix CSV dataframe by processing POST_IDs and removing invalid rows.
    
    Args:
        df: Dataframe to clean
    
    Returns:
        pandas.DataFrame: Cleaned dataframe with standardized POST_IDs and no duplicates
    """
    df['POST_ID'] = df['POST_ID'].astype(str).str[:6]
    df = df.dropna(how='all')
    df = df.drop_duplicates().reset_index(drop=True)
    df['POST_ID'].replace('', pd.NA, inplace=True)
    df = df.dropna(subset=['POST_ID']).reset_index(drop=True)
    return df

def merge_csvs(df1, df2):
    """
    Merge two dataframes by concatenating them.
    
    Args:
        df1: First dataframe to merge
        df2: Second dataframe to merge
    
    Returns:
        pandas.DataFrame: Merged dataframe with combined rows from both dataframes
    """
    result = pd.concat([df1, df2], ignore_index=True)
    return result

def save_enhanced_df(enhanced_df):
    """
    Save enhanced dataframe to both pickle and CSV formats.
    
    Args:
        enhanced_df: Dataframe to save
    """
    enhanced_df.to_pickle(os.path.join(DATA_FOLDER, "enhanced_reddit_df.pkl"))
    enhanced_df.to_csv(os.path.join(DATA_FOLDER, "enhanced_reddit_df.csv"))


if __name__ == "__main__":
    df1 = load_file(os.path.join(DATA_FOLDER, 'reddit_enhanced_backup_2.csv'))
    df2 = load_file(os.path.join(DATA_FOLDER, 'reddit_enhanced_backup_3.csv'))
    df1 = fix_csv(df1)
    df2 = fix_csv(df2)
    enhanced_df = merge_csvs(df1, df2)
    save_enhanced_df(enhanced_df)