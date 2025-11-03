import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_utils import load_enhanced_df
import pandas as pd

def draw_violin_plots():    
    df = load_enhanced_df()

    df.loc[df["LINK_SENTIMENT"] == -1, "LINK_SENTIMENT"] = 0
    df["LINK_SENTIMENT"] = df["LINK_SENTIMENT"].astype(int)

    df['Sentiment_compound_vader'] = pd.to_numeric(df['Sentiment_compound_vader'], errors='coerce')
    body_df = df[df["source_type"] == "body"]
    title_df = df[df["source_type"] == "title"]


    _, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.violinplot(
        data=body_df,
        x='LINK_SENTIMENT', 
        y='Sentiment_compound_vader',
        inner='box',
        ax=axes[0]
    )
    axes[0].set_title('Body: VADER Sentiment by Link Sentiment')
    axes[0].set_xlabel('Link Sentiment')
    axes[0].set_ylabel('VADER Compound')

    sns.violinplot(
        data=title_df,
        x='LINK_SENTIMENT', 
        y='Sentiment_compound_vader',
        inner='box',
        ax=axes[1]
    )
    axes[1].set_title('Title: VADER Sentiment by Link Sentiment')
    axes[1].set_xlabel('Link Sentiment')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()