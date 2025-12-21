import os
from src.utils.data_utils import load_final_df


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'cluster_visualization')


def export_csv(df, subreddits_per_cluster=3):
    df_reduced = df[['source_supercluster', 'source_cluster_level_1']]
    df_reduced = df_reduced.drop_duplicates()

    source_subreddits = set()

    for _, row in df_reduced.iterrows():
        df_subreddits = df[
            (df['source_supercluster'] == row['source_supercluster']) &
            (df['source_cluster_level_1'] == row['source_cluster_level_1'])
        ].groupby('SOURCE_SUBREDDIT').agg({'POST_ID': 'count'}).reset_index()
        df_subreddits.columns = ['SOURCE_SUBREDDIT', 'POST_ID_COUNT']
        df_subreddits = df_subreddits.sort_values(by='POST_ID_COUNT', ascending=False)
        df_subreddits = df_subreddits.head(subreddits_per_cluster)
        source_subreddits.update(df_subreddits['SOURCE_SUBREDDIT'].tolist())

    df_posts = df[df['SOURCE_SUBREDDIT'].isin(source_subreddits)][['source_supercluster', 'source_cluster_level_1', 'SOURCE_SUBREDDIT']]
    df_posts = df_posts.drop_duplicates()
    df_posts['SOURCE_SUBREDDIT'] = df_posts['SOURCE_SUBREDDIT'].apply(lambda x: "r/" + x.lower())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_posts.to_csv(os.path.join(OUTPUT_DIR, 'df_posts.csv'), index=False)


if __name__ == "__main__":
    df = load_final_df()
    export_csv(df, subreddits_per_cluster=4)
