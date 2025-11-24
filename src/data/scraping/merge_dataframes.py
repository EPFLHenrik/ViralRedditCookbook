import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_utils import load_reddit_df, load_enhanced_df


def merge_dataframes(reddit_df_merged_path='reddit_df_merged.pkl', enhanced_df_path='enhanced_reddit_df_finalized.pkl', final_df_path='reddit_df_final.pkl'):
    """
    Merge the reddit_df_merged.pkl and enhanced_reddit_df_finalized.pkl into a final dataframe.
    
    Args:
        reddit_df_merged_path: Path to the reddit_df_merged.pkl file.
        enhanced_df_path: Path to the enhanced_reddit_df_finalized.pkl file.
        final_df_path: Path to the reddit_df_final.pkl file.
    """

    
    df = load_reddit_df(os.path.join(DATA_FOLDER, reddit_df_merged_path))
    df_enh = load_enhanced_df(os.path.join(DATA_FOLDER, enhanced_df_path))

    df["ups"] = df_enh["ups"]
    df = df.dropna(subset=['ups'])

    output_file = os.path.join(DATA_FOLDER, final_df_path)
    with open(output_file, "wb") as f:
        df.to_pickle(f)
