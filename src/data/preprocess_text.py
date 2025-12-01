
def preporcess_text(df):
    """
    Preprocess the text of the dataframe. Removing deleted and removed posts.
    
    Args:
        df: The dataframe to preprocess.
    
    Returns:
        The preprocessed dataframe.
    """

    df = df.dropna(subset=['title', 'selftext'])
    df = df[~df['title'].str.contains(r"\[deleted by user\]|\[removed\]", case=False, na=False)]
    df = df[~df['selftext'].str.contains(r"\[deleted by user\]|\[removed\]", case=False, na=False)]
    return df
