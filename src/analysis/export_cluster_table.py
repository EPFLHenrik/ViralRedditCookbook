import os
import json
import pickle
import pandas as pd
import numpy as np

from src.utils.data_utils import load_reddit_graph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'clustering_results')

print("Reading Reddit graph…")
G = load_reddit_graph()
print(f"graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

with open(os.path.join(RESULTS_DIR, "labels_per_level.pkl"), "rb") as f:
    labels_per_level = pickle.load(f)

with open(os.path.join(RESULTS_DIR, "exemplars.json"), "r") as f:
    exemplars = json.load(f)

embeddings_npz = os.path.join(RESULTS_DIR, "processed_embeddings.npz")
data = dict(np.load(embeddings_npz, allow_pickle=True))
subreddit_names = data["subreddit_names"]

rows = []
for i, name in enumerate(subreddit_names):
    row = {"subreddit_name": name}
    degree = int(G.degree(name)) if name in G else 0
    if degree < 1000:
        continue
    row["num_outgoing_edges"] = degree

    for lvl in sorted(labels_per_level.keys()):
        cluster_id = labels_per_level[lvl].get(i, None)
        if cluster_id is None:
            value = name
        else:
            last_part = str(cluster_id).split(".")[-1]
            try:
                idx = int(last_part)
                letters = ""
                while True:
                    idx, rem = divmod(idx, 26)
                    letters = chr(65 + rem) + letters
                    if idx == 0:
                        break
                value = letters
            except ValueError:
                value = str(cluster_id)
        row[f"level_{lvl}_cluster"] = value

    rows.append(row)

df = pd.DataFrame(rows)
print(f"rows: {len(df)}")

output_path = os.path.join(RESULTS_DIR, "subreddit_cluster_assignments.csv")
df.to_csv(output_path, index=False)
print(f"wrote {output_path}")
