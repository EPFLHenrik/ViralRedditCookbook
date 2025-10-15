import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def preprocess_dataframe(load_type='body', body_tsv_file='soc-redditHyperlinks-body.tsv', 
                     title_tsv_file='soc-redditHyperlinks-title.tsv', force_reload=False):

    data_folder = DATA_DIR

    if load_type == 'body':
        tsv_file = os.path.join(data_folder, body_tsv_file)
        pickle_file = os.path.join(data_folder, 'reddit_df_body.pkl')
    elif load_type == 'title':
        tsv_file = os.path.join(data_folder, title_tsv_file)
        pickle_file = os.path.join(data_folder, 'reddit_df_title.pkl')
    elif load_type == 'merged':
        tsv_file = [os.path.join(data_folder, body_tsv_file), os.path.join(data_folder, title_tsv_file)]
        pickle_file = os.path.join(data_folder, 'reddit_df_merged.pkl')
    else:
        raise ValueError(f"Invalid load_type: {load_type}. Must be 'body', 'title', or 'merged'")
    
    if not force_reload and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            df = pd.read_pickle(f)
            return df

    if load_type == 'merged':
        body_df = pd.read_csv(tsv_file[0], sep='\t')
        title_df = pd.read_csv(tsv_file[1], sep='\t')
        
        body_df['source_type'] = 'body'
        title_df['source_type'] = 'title'
        
        df = pd.concat([body_df, title_df], ignore_index=True)
    else:
        df = pd.read_csv(tsv_file, sep='\t')

    properties_names = [
        "Num_characters", "Num_characters_no_space", "Frac_alpha", "Frac_digits", "Frac_uppercase", "Frac_whitespace", "Frac_special", "Num_words", "Num_unique_words", "Num_long_words", "Avg_word_length", "Num_unique_stopwords", "Frac_stopwords", "Num_sentences", "Num_long_sentences", "Avg_chars_per_sentence", "Avg_words_per_sentence", "Automated_readability_index", "Sentiment_pos_vader", "Sentiment_neg_vader", "Sentiment_compound_vader", "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", "LIWC_Friends", "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"
    ]
    properties_df = df['PROPERTIES'].str.split(',', expand=True)
    properties_df.columns = properties_names
    df = pd.concat([df.drop(columns=['PROPERTIES']), properties_df], axis=1)

    df.to_pickle(pickle_file)
    return df

if __name__ == "__main__":
    df = preprocess_dataframe(load_type='merged',
                            body_tsv_file='soc-redditHyperlinks-body.tsv',
                            title_tsv_file='soc-redditHyperlinks-title.tsv')

    print("Data frame loaded successfully, number of lines: ", df.shape[0], "number of columns: ", df.shape[1])
