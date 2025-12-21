from src.utils.data_utils import load_enhanced_df,load_final_df
from src.analysis.virality_analysis.get_viral_posts import get_viral_posts, plot_viral_posts
import pandas as pd
import networkx as nx
from typing import Union
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import spearmanr, pearsonr


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "src", "analysis")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def edges_negative_by_quarter(G: nx.MultiDiGraph, 
                               quarter_end: str = "DEC", 
                               save_csv: bool = True, 
                               output_dir: str = ANALYSIS_DIR, 
                               filename: str = "reddit_negative_edges_by_quarter.csv") -> pd.DataFrame:
    
    """
    Extract edges with negative sentiment and index them by 3-month windows (calendar quarters),
    then optionally save to a CSV.

    Parameters
    
    G : nx.MultiDiGraph
        Graph produced by preprocess_reddit_graph.
    quarter_end : {"DEC","JAN","FEB",...}
        Anchor month for quarter end. "DEC" = standard Q1=Jan–Mar.
    save_csv : bool
        If True, save the result to a CSV.
    output_dir : str or None
        Directory to save the CSV in. If None, saves in the same folder as the data directory.
    filename : str
        CSV file name.

    Returns
    
    pandas.DataFrame
        Columns: ['timestamp']
        Index: MultiIndex [window, source, target]
    """
    rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("sentiment") == -1:
            ts = pd.to_datetime(d.get("timestamp"),format='mixed')
            if pd.notna(ts):
                rows.append({"source": u, "target": v, "timestamp": ts})

    df = pd.DataFrame(rows)
    if df.empty:
        print(" No edges with sentiment = -1 found.")
        mi = pd.MultiIndex.from_arrays(
            [pd.PeriodIndex([], freq=f"Q-{quarter_end}"), [], []],
            names=["window", "source", "target"]
        )
        empty_df = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")}, index=mi)
        return empty_df
    
    df["window"] = df["timestamp"].dt.to_period(f"Q-{quarter_end}")
    out = (df.set_index(["window", "target","source"])
             .sort_index()[["timestamp"]])
   
    if save_csv:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, filename)
       
        out.reset_index().to_csv(csv_path, index=False)
        print(f"Saved negative edges CSV to: {csv_path}")

    return out

def sources_by_quarter_matrix(
    data: Union[str, pd.DataFrame],
    quarter_end: str = "DEC",
    count_distinct_targets: bool = True,  
    ignore_quarters: list = ["2013Q4"],   
    save_csv: bool = True,
    output_path: str = os.path.join(ANALYSIS_DIR, "targets_num_sources_by_window.csv"),
) -> pd.DataFrame:
    
    """
    Build a wide matrix with sources as rows and quarters as columns.
    Keep only sources that appear in *every* quarter (after ignoring specified ones).

    Input data must have columns: ['window','target','source','timestamp'].
    'window' can be strings like '2014Q1' and will be coerced to Periods (Q-<quarter_end>).

    Parameters
    
    data : str | pd.DataFrame
        CSV path or DataFrame with window/target/source/timestamp
    quarter_end : str
        Quarter anchor (e.g. 'DEC' for standard calendar quarters)
    count_distinct_targets : bool
        If True, count distinct targets per quarter; else count raw edges
    ignore_quarters : list of str
        Quarters to exclude from the analysis (e.g. ["2013Q4"])
    save_csv : bool
        Save the wide table to disk
    output_path : str
        CSV destination path

    Returns
    
    pd.DataFrame
        Index: source
        Columns: quarters (YYYYQn)
        Values: per-quarter counts
    """
    if isinstance(data, str):
        df = pd.read_csv(data, parse_dates=["timestamp"])
    else:
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"],format='mixed')

    if not isinstance(df["window"].dtype, pd.PeriodDtype):
        df["window"] = pd.PeriodIndex(df["window"].astype(str), freq=f"Q-{quarter_end}")

    ignore_periods = set(pd.Period(q, freq=f"Q-{quarter_end}") for q in ignore_quarters)
    df = df[~df["window"].isin(ignore_periods)].copy()

    if count_distinct_targets:
        agg = (
            df.loc[:, ["window", "source", "target"]]
              .drop_duplicates()
              .groupby(["window", "source"], observed=True)["target"]
              .nunique()
              .rename("value")
              .reset_index()
        )
    else:
        agg = (
            df.groupby(["window", "source"], observed=True)
              .size()
              .rename("value")
              .reset_index()
        )
    
    wide = (agg.pivot(index="source", columns="window", values="value")
              .fillna(0)
              .astype(int))

    wide = wide.reindex(sorted(wide.columns), axis=1)
    wide.columns = [f"{p.year}Q{p.quarter}" for p in wide.columns]

    wide = wide[wide.gt(0).all(axis=1)]

    if save_csv:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wide.to_csv(output_path, index=True)

    return wide

def corr_liwc_anger_vs_ups():

    """Calculate and display Pearson and Spearman correlations between post anger (LIWC_Anger) and upvotes."""

    df = load_enhanced_df()

    df['LIWC_Anger'] = pd.to_numeric(df['LIWC_Anger'], errors='coerce')
    df['ups'] = pd.to_numeric(df['ups'], errors='coerce')

    df = df.dropna(subset=['LIWC_Anger', 'ups']).copy()

    anger = df['LIWC_Anger'].values
    ups = df['ups'].values

    rho_spear, p_spear = spearmanr(anger, ups)

    rho_pear, p_pear = pearsonr(anger, ups)

    print("Correlation between LIWC_Anger and Upvotes:")
    print(f"  Spearman  ρ = {rho_spear:.3f}, p = {p_spear:.3e}")
    print(f"  Pearson   r = {rho_pear:.3f}, p = {p_pear:.3e}")

def corr_vader_neg_vs_ups():
    
    """Calculate and display Pearson and Spearman correlations between VADER negative sentiment and upvotes."""

    df = load_enhanced_df()

    df['Sentiment_neg_vader'] = pd.to_numeric(df['Sentiment_neg_vader'], errors='coerce')
    df['ups'] = pd.to_numeric(df['ups'], errors='coerce')

    df = df.dropna(subset=['Sentiment_neg_vader', 'ups']).copy()
    
    anger = df['Sentiment_neg_vader'].values
    ups = df['ups'].values

    rho_spear, p_spear = spearmanr(anger, ups)

    rho_pear, p_pear = pearsonr(anger, ups)

    print("Correlation between Sentiment_neg_vader and Upvotes:")
    print(f"  Spearman  ρ = {rho_spear:.3f}, p = {p_spear:.3e}")
    print(f"  Pearson   r = {rho_pear:.3f}, p = {p_pear:.3e}")
   
def corr_time_vs_ups(): 

    """Spearman correlations between time and upvotes (post-level and monthly means)."""

    df=load_enhanced_df()

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format='mixed')
    df = df.dropna(subset=['TIMESTAMP', 'ups'])

    df['ups'] = pd.to_numeric(df['ups'], errors='coerce')
    df = df.dropna(subset=['ups']).copy()

    t_numeric = df['TIMESTAMP'].astype('int64') // 10**9
    ups = df['ups'].values

    rho_post, p_post = spearmanr(t_numeric, ups)
    
    df['year_month'] = df['TIMESTAMP'].dt.to_period('M')
    monthly = (df.groupby('year_month')['ups']
                 .mean()
                 .reset_index())

    month_index = np.arange(len(monthly))
    rho_month, p_month = spearmanr(month_index, monthly['ups'].values)
    

    print("Post-level correlation (time ↔ ups):")
    print(f"  Spearman  ρ = {rho_post:.3f}, p = {p_post:.3e}")
    

    print("\nMonthly-mean correlation (time ↔ avg ups):")
    print(f"  Spearman  ρ = {rho_month:.3f}, p = {p_month:.3e}")


def edges_all_by_quarter(
    quarter_end: str = "DEC",
    save_csv: bool = True,
    output_dir: str = ANALYSIS_DIR, 
    ) -> pd.DataFrame:

    """Aggregate upvotes by quarter and source subreddit,
      split by link sentiment (+1/−1), and optionally save CSVs.

      Parameters
    
    quarter_end : str, optional
        The anchor month for the quarter-end period. 
    save_csv : bool, optional
        If True, saves two CSV files:
    output_dir : str, optional
        Directory where the CSV files will be saved.
        """
    
    df=load_enhanced_df()
    
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format='mixed')

    df["WINDOW"] = df["TIMESTAMP"].dt.to_period(f"Q-{quarter_end}")

    df = df.dropna(subset=["TIMESTAMP", "ups"])

    df_pos = df[df["LINK_SENTIMENT"] == 1]
    df_neg = df[df["LINK_SENTIMENT"] == -1]

    out_pos = (df_pos.set_index(["WINDOW", "SOURCE_SUBREDDIT"])
             .sort_index()[['ups']])
    out_neg=(df_neg.set_index(["WINDOW", "SOURCE_SUBREDDIT"])
             .sort_index()[['ups']])

    if save_csv:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(output_dir, exist_ok=True)

        csv_path_pos = os.path.join(output_dir, "reddit_pos_edges_by_quarter.csv")
        csv_path_neg = os.path.join(output_dir,"reddit_neg_edges_by_quarter.csv")
       
        out_pos.reset_index().to_csv(csv_path_pos, index=False)
        out_neg.reset_index().to_csv(csv_path_neg, index=False)


        print(f"Saved all positive edges CSV to: {csv_path_pos}")
        print(f"Saved all negative edges CSV to: {csv_path_neg}")



def mean_ups_matrix_by_quarter(
    csv_path: str = os.path.join(ANALYSIS_DIR, "reddit_pos_edges_by_quarter.csv"),
    fill_missing: float = 0.0,
    save_csv: bool = True,
    output_dir: str = ANALYSIS_DIR,
    filename: str = "mean_pos_ups_by_quarter.csv"
) -> pd.DataFrame:
    """Build a subreddit × quarter matrix of mean upvotes,
      fill missing cells, and optionally save as CSV.
      
        Parameters
    
    csv_path : str, optional
        Path to the input CSV file.
    fill_missing : float, optional
        Value used to fill missing entries (e.g., when a subreddit has no data for a quarter).
    save_csv : bool, optional
        If True, saves the resulting subreddit × quarter matrix to CSV.
    output_dir : str, optional
        Directory in which to save the output CSV. 
    filename : str, optional
        Output CSV filename.
        
    """
    
    df = pd.read_csv(csv_path)

    df["ups"] = pd.to_numeric(df["ups"], errors="coerce")
    df = df.dropna(subset=["WINDOW", "SOURCE_SUBREDDIT", "ups"])

    grp = df.groupby(["SOURCE_SUBREDDIT", "WINDOW"], as_index=False)["ups"].mean()

    mat = grp.pivot(index="SOURCE_SUBREDDIT", columns="WINDOW", values="ups")

    try:
        periods = pd.PeriodIndex(mat.columns, freq="Q-DEC")
        mat = mat.reindex(columns=periods.sort_values().astype(str))
    except Exception:
        mat = mat.reindex(sorted(mat.columns), axis=1)

    mat = mat.fillna(fill_missing)

    if save_csv:
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)
        mat.to_csv(output_path)
        print(f"Saved mean ups matrix to: {output_path}")

    return mat

def load_and_prepare_upvotes():
    """
    Load Reddit data and extract relevant temporal features for upvote analysis.
    
    """
    df = load_enhanced_df()

    df = df.dropna(subset=["TIMESTAMP", "ups"])
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format='mixed')
    df["ups"] = pd.to_numeric(df["ups"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP", "ups"])

    df["year_month"] = df["TIMESTAMP"].dt.to_period("M")
    df["hour"] = df["TIMESTAMP"].dt.hour
    df["dayofweek"] = df["TIMESTAMP"].dt.day_name()
    df["month"] = df["TIMESTAMP"].dt.month

    return df

def plot_monthlymean_upvotes(df):
    """
    Plot the evolution of average upvotes over time (by month).

    """
    monthly_mean = df.groupby("year_month", as_index=False)["ups"].mean()
    x_labels = monthly_mean["year_month"].astype(str).tolist()
    y = monthly_mean["ups"].to_numpy()
    x = np.arange(len(x_labels))

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.xticks(x[::3], x_labels[::3], rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Average upvotes")
    plt.title("Evolution of Average Upvotes Over Time")
    plt.tight_layout()
    plt.show()

def plot_weekday_upvotes(df):
    """
    Plot average upvotes by day of the week.
    
    """
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_mean = df.groupby("dayofweek")["ups"].mean().reindex(dow_order)
    plt.figure(figsize=(10, 5))
    dow_mean.plot(kind="bar", color="lightgreen")
    plt.title("Average Upvotes by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Mean Upvotes")
    plt.tight_layout()
    plt.show()

def plot_monthly_upvotes(df):
    """
    Plot average upvotes by calendar month across all years.
    
    """
    month_mean = df.groupby("month")["ups"].mean().reindex(range(1,13))
    plt.figure(figsize=(10, 5))
    month_mean.plot(kind="bar", color="salmon")
    plt.title("Average Upvotes by Month (Across Years)")
    plt.xlabel("Month (1–12)")
    plt.ylabel("Mean Upvotes")
    plt.tight_layout()
    plt.show()
    
def getviralpoststemplate(virality_percentile=0.9997): 
    """
    Build per-supercluster CSV files combining:
    - sparse moving-average virality (for trend lines)
    - individual viral posts 

    """
    final_by_supercluster = {}
    dfs_by_supercluster = {}
    subsampledict={"Gaming & Interactive Entertainment": 500, 
               "Lifestyle & Niche Interests": 200, 
               "Politics & Society": 500 ,
               "Pop Culture & Media": 500, 
               "Reddit Meta & Community":1000 ,
               "Sports & Athletics" : 200 , 
               "Technology & Digital Culture": 300}
    df = load_final_df(calculate_rss=True)
    df = get_viral_posts(df, virality_percentile=virality_percentile, window_size='60D')

    for supercluster, df_sc in df.groupby("source_supercluster"):
        df_sc = df_sc.copy()
        df_sc = df_sc.sort_values("TIMESTAMP")
        dfs_by_supercluster[supercluster] = df_sc
        df_v = df_sc[df_sc["is_viral"]].copy()
        df_v["Info"] = df_v.apply(
        lambda row: (
            f"<b>Source Subreddit :</b> {row['SOURCE_SUBREDDIT']}<br>"
            f"<b>Target Subreddit :</b> {row['TARGET_SUBREDDIT']}<br>"
            f"<b>Title :</b> {row['title']}<br>"
            # f"<b>Text :</b> {row['selftext']}<br>"
            f"<b>Virality_rss :</b> {row['virality_rss']:.3f}<br>"
            f"<b>Timestamp :</b> {row['TIMESTAMP']}"
        ) ,
        axis=1 )

        df_v["color"]=df_v["SOURCE_SUBREDDIT"]
        df_v["connect"]=""
        df_v["size"]=20
        df_v["name"]="Viral Post"
        df_v=df_v[["TIMESTAMP","virality_rss","color","size","connect","Info","name"]]
        
        df_avg = df_sc[["TIMESTAMP", "moving_avg_virality"]].copy()

        df_avg = df_avg.iloc[::subsampledict[supercluster]]    
        df_avg = df_avg.iloc[5:]
        df_avg["Info"]=df_avg.apply( lambda row : (f"<b>Virality_rss MA :</b> {row['moving_avg_virality']:.3f}<br>"
                                                f"<b>Timestamp :</b> {row['TIMESTAMP']}"
                                                    ), axis=1 )
        df_avg["color"]="same color"
        df_avg["connect"]="connect line"
        df_avg["size"]=2
        df_avg["name"]="Moving Average"
        df_avg = df_avg.rename(columns={"moving_avg_virality": "virality_rss"})
        
        df_final_cluster = pd.concat([df_avg, df_v], ignore_index=True)

        final_by_supercluster[supercluster] = df_final_cluster
    for supercluster, df_sc in dfs_by_supercluster.items():
        n_true = df_sc["is_viral"].sum()   
        print(supercluster, n_true)

    for supercluster, df_final_cluster in final_by_supercluster.items():
        df_final_cluster["TIMESTAMP"] = pd.to_datetime(df_final_cluster["TIMESTAMP"]).dt.date
        filename = f"final_supercluster_{supercluster}.csv" 
        df_final_cluster.to_csv(os.path.join(DATA_DIR, filename), index=False)
def getmeanupsracetemplate(month="DEC"):
    """
    Build a supercluster × quarter matrix of mean upvotes and save it as CSV.
    """
    df = load_final_df(calculate_rss=False)
    df=df[["source_supercluster","TIMESTAMP","ups"]] 
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format='mixed')
    df["WINDOW"] = df["TIMESTAMP"].dt.to_period(f"Q-{month}")
    df = df.dropna(subset=["TIMESTAMP", "ups"]) 
    df=df.groupby(["source_supercluster","WINDOW"])["ups"].mean().reset_index()
    mat = df.pivot(index="source_supercluster", columns="WINDOW", values="ups")
    try:
        periods = pd.PeriodIndex(mat.columns, freq="Q-DEC")
        mat = mat.reindex(columns=periods.sort_values().astype(str))
    except Exception:
        mat = mat.reindex(sorted(mat.columns), axis=1)
        mat = mat.fillna(0)
    filename="ClustersMeanUpsRace.csv"
    mat.to_csv(os.path.join(DATA_DIR, filename), index=True)


def getviralityGeomRacetemplate(month="DEC"): 
    """
    Build a supercluster × quarter matrix of geometric-mean virality
    (positive virality only) and save it as CSV.
    
    """
    df= load_final_df(calculate_rss=True)
    df=get_viral_posts(df)
    df=df[["source_supercluster","TIMESTAMP","virality_rss"]]
    df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"],format='mixed')
    df["WINDOW"] = df["TIMESTAMP"].dt.to_period(f"Q-{month}") 
    df = df.dropna(subset=["TIMESTAMP", "virality_rss"]) 
    df = df[df["virality_rss"] > 0]
    df_geo = ( df.groupby(["source_supercluster", "WINDOW"])["virality_rss"] .apply(lambda x: np.exp(np.mean(np.log(x)))) .reset_index(name="virality_rss_gmean") ) 
    mat = df_geo.pivot(index="source_supercluster", columns="WINDOW", values="virality_rss_gmean") 
    try: 
        periods = pd.PeriodIndex(mat.columns, freq="Q-DEC") 
        mat = mat.reindex(columns=periods.sort_values().astype(str)) 
    except Exception: 
        mat = mat.reindex(sorted(mat.columns), axis=1) 
        mat = mat.fillna(0) 
    mat = mat.drop(columns=mat.columns[0])
    filename="ClustersViralityRacegeometric.csv" 
    mat.to_csv(os.path.join(DATA_DIR, filename), index=True)
    





    

# if __name__ == "__main__":
    # G = load_reddit_graph()
    # sources_by_quarter_matrix(r"reddit_negative_edges_by_quarter.csv")
    # corr_time_vs_ups()
    # corr_liwc_anger_vs_ups()
    # corr_vader_neg_vs_ups()
    # edges_all_by_quarter()
    # mean_ups_matrix_by_quarter()
    # corr_time_vs_ups()
    # getmeanupsracetemplate()
    # getviralpoststemplate()
    # getviralityracetemplate()