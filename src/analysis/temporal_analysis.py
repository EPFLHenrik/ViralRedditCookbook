from src.utils.data_utils import load_reddit_graph
import pandas as pd
import networkx as nx
from typing import Union
import os

G = load_reddit_graph()

for u, v, data in list(G.edges(data=True))[:10]:
    print(f"{u} -> {v}, attributes = {data}")

def edges_negative_by_quarter(G: nx.MultiDiGraph, 
                               quarter_end: str = "DEC", 
                               save_csv: bool = True, 
                               output_dir: str = r"C:\Users\Aziz Hamza\Desktop\ADAMA1\ada-2025-project-standard-deviants-ada\src\analysis", 
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
    output_path: str = r"C:\Users\Aziz Hamza\Desktop\ADAMA1\ada-2025-project-standard-deviants-ada\src\analysis\targets_num_sources_by_window.csv",
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
        if not output_path:
            raise ValueError("Provide output_path when save_csv=True.")
        wide.to_csv(output_path, index=True)

    return wide
if __name__ == "__main__":
    # G = load_reddit_graph()
    sources_by_quarter_matrix(r"C:\Users\Aziz Hamza\Desktop\ADAMA1\ada-2025-project-standard-deviants-ada\src\analysis\reddit_negative_edges_by_quarter.csv")

    
