import pandas as pd


def remove_non_ascii(text):
    """
    Remove all non-ASCII characters from text.
    """
    if pd.isna(text):
        return text
    text_str = str(text)
    ascii_text = ''.join(c for c in text_str if ord(c) < 128)
    return ascii_text

def preporcess_text(df):
    """
    Preprocess the text of the dataframe. Removing deleted and removed posts.
    
    Args:
        df: The dataframe to preprocess.
    
    Returns:
        The preprocessed dataframe.
    """

    df = df.dropna(subset=['title', 'selftext'])
    df = df[~df['title'].str.contains(r"\[deleted\]|\[deleted by user\]|\[removed\]", case=False)]
    df = df[~df['selftext'].str.contains(r"\[deleted\]|\[deleted by user\]|\[removed\]", case=False)]

    df['title'] = df['title'].apply(remove_non_ascii)
    df['selftext'] = df['selftext'].apply(remove_non_ascii)

    return df
