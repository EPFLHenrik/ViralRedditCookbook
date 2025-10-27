from src.utils.data_utils import load_reddit_graph
import pandas as pd
import networkx as nx
from typing import Union
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import spearmanr, kendalltau

# G = load_reddit_graph()

# for u, v, data in list(G.edges(data=True))[:10]:
#     print(f"{u} -> {v}, attributes = {data}")

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
    ----------
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
    -------
    pandas.DataFrame
        Columns: ['timestamp']
        Index: MultiIndex [window, source, target]
    """
    rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if d.get("sentiment") == -1:
            ts = pd.to_datetime(d.get("timestamp"), errors="coerce", utc=True)
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
    ----------
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
    -------
    pd.DataFrame
        Index: source
        Columns: quarters (YYYYQn)
        Values: per-quarter counts
    """
    # --- Load ---
    if isinstance(data, str):
        df = pd.read_csv(data, parse_dates=["timestamp"])
    else:
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

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

def plotupvotes(): 
    file_path = os.path.join(DATA_DIR, "reddit_enhanced_backup_3.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path,usecols=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT','TIMESTAMP', 'ups'],parse_dates=["TIMESTAMP"],        # <- important
        low_memory=False)
    df = df.dropna()
    df.loc[:, 'ups'] = df['ups'].astype(int)
    df.loc[:, 'TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    df.loc[:, 'year_month'] = df['TIMESTAMP'].dt.to_period('M')
    df.loc[:, 'hour'] = df['TIMESTAMP'].dt.hour
    df.loc[:,'dayofweek'] = df['TIMESTAMP'].dt.day_name()
    monthly_mean = df.groupby('year_month')['ups'].mean().reset_index()
    plt.figure(figsize=(10,5))
    plt.plot(monthly_mean['year_month'].astype(str), monthly_mean['ups'])
    plt.xticks(monthly_mean['year_month'].astype(str)[::3], rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Average upvotes")
    plt.title("Evolution of average upvotes over time")
    plt.show()

    hourly_mean = df.groupby('hour')['ups'].mean()
    plt.figure(figsize=(10,5))
    hourly_mean.plot(kind='bar')
    plt.title("Average Upvotes by Hour of Day")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Mean Upvotes")
    plt.show()

    dow_mean = df.groupby('dayofweek')['ups'].mean().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    plt.figure(figsize=(10,5))
    dow_mean.plot(kind='bar', color='skyblue')
    plt.title("Average Upvotes by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Mean Upvotes")
    plt.show()
    df['month'] = df['TIMESTAMP'].dt.month
    month_mean = df.groupby('month')['ups'].mean()
    plt.figure(figsize=(10,5))
    month_mean.plot(kind='bar')
    plt.title("Average Upvotes by Month")
    plt.xlabel("Month (1–12)")
    plt.ylabel("Mean Upvotes")
    plt.show()

def corr_time_vs_ups(): 
    file_path = os.path.join(DATA_DIR, "reddit_enhanced_backup_3.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path,usecols=['TIMESTAMP', 'ups'],parse_dates=["TIMESTAMP"], low_memory=False)
    df = df.dropna(subset=['TIMESTAMP', 'ups']).copy()
    df['ups'] = pd.to_numeric(df['ups'], errors='coerce')
    df = df.dropna(subset=['ups']).copy()
    t_numeric = df['TIMESTAMP'].astype('int64') // 10**9
    ups = df['ups'].values
    # post level corr
    rho_post, p_post = spearmanr(t_numeric, ups)
    

    # Monthly-aggregated correlation ---
    df['year_month'] = df['TIMESTAMP'].dt.to_period('M')
    monthly = (df.groupby('year_month')['ups']
                 .mean()
                 .reset_index())
    # Use month index as numeric time
    month_index = np.arange(len(monthly))
    rho_month, p_month = spearmanr(month_index, monthly['ups'].values)
    

    print("Post-level correlation (time ↔ ups):")
    print(f"  Spearman  ρ = {rho_post:.3f}, p = {p_post:.3e}")
    

    print("\nMonthly-mean correlation (time ↔ avg ups):")
    print(f"  Spearman  ρ = {rho_month:.3f}, p = {p_month:.3e}")

    

if __name__ == "__main__":
    # G = load_reddit_graph()
    # sources_by_quarter_matrix(r"C:\Users\Aziz Hamza\Desktop\ADAMA1\ada-2025-project-standard-deviants-ada\src\analysis\reddit_negative_edges_by_quarter.csv")
    # plotupvotes()
    corr_time_vs_ups()
    
