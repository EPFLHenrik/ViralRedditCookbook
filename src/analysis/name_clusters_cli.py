import os
import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

from src.utils.data_utils import load_reddit_graph

RESULTS_DIR = r"src/analysis/clustering_results"
PKL_IN  = os.path.join(RESULTS_DIR, "labels_per_level.pkl")
PKL_OUT = os.path.join(RESULTS_DIR, "labels_per_level_named.pkl")
CSV_OUT = os.path.join(RESULTS_DIR, "subreddit_cluster_assignments_named.csv")
PROMPTS_DIR = os.path.join(RESULTS_DIR, "cluster_naming_prompts")
EMBEDDINGS_FILE = os.path.join(RESULTS_DIR, "processed_embeddings.npz")

MIN_EDGE_COUNT = 1000
MIN_QUALIFIED = 5
MAX_REPRESENTATIVES = 20

os.makedirs(PROMPTS_DIR, exist_ok=True)

def load_all_data():
    G = load_reddit_graph()
    with open(PKL_IN, "rb") as f:
        labels_per_level = pickle.load(f)
    data = dict(np.load(EMBEDDINGS_FILE, allow_pickle=True))
    subreddit_names = data["subreddit_names"]

    node_info = {}
    for i, name in enumerate(subreddit_names):
        out_degree = G.out_degree(name) if name in G else 0
        node_info[i] = (name, out_degree)

    return G, labels_per_level, subreddit_names, node_info


def build_cluster_membership(labels_per_level, node_info):
    cluster_members = {}
    for level in sorted(labels_per_level.keys()):
        level_clusters = defaultdict(list)
        for node_idx, cluster_id in labels_per_level[level].items():
            if node_idx in node_info:
                name, out_degree = node_info[node_idx]
                level_clusters[cluster_id].append((node_idx, name, out_degree))
        cluster_members[level] = dict(level_clusters)
    return cluster_members


def get_cluster_representatives(cluster_members_list):
    sorted_members = sorted(cluster_members_list, key=lambda x: x[2], reverse=True)
    qualified = [m for m in sorted_members if m[2] >= MIN_EDGE_COUNT]

    if len(cluster_members_list) == 1:
        return None

    if len(qualified) < MIN_QUALIFIED:
        n = min(5, len(sorted_members))
        representatives = [name for _, name, _ in sorted_members[:n]]
    else:
        n = min(MAX_REPRESENTATIVES, len(qualified))
        representatives = [name for _, name, _ in qualified[:n]]

    return representatives


def cluster_id_to_path_key(cluster_id, level):
    parts = str(cluster_id).split(".")
    parts = parts[: level + 1]

    letters = []
    for part in parts:
        try:
            idx = int(part)
            letter = ""
            while True:
                idx, rem = divmod(idx, 26)
                letter = chr(65 + rem) + letter
                if idx == 0:
                    break
            letters.append(letter)
        except ValueError:
            letters.append(str(part))

    return "|".join(letters)


def path_key_to_cluster_id(path_key, cluster_id_to_path, level):
    for cid, pkey in cluster_id_to_path[level].items():
        if pkey == path_key:
            return cid
    return None


def make_prompt(level, cluster_representatives, cluster_id_to_path):
    header = (
        "You are an expert at naming subreddit communities.\n"
        f"Task: For each cluster PATH key at level {level}, return a short, clear title (2–4 words max).\n"
        "The PATH key encodes the full hierarchy (e.g., A|C|B).\n"
        "Rules:\n"
        "1) Return *only* valid JSON mapping PATH -> Title.\n"
        "2) Keep titles concise and specific, avoid emojis and punctuation.\n"
        "3) If unsure, make your best guess based on the representative subreddits.\n\n"
        "Input clusters (PATH -> list of representative subreddits):\n"
    )

    payload = {}
    for cluster_id, representatives in cluster_representatives.items():
        if representatives is None:
            continue
        path_key = cluster_id_to_path[level][cluster_id]
        payload[path_key] = representatives

    if not payload:
        return None

    input_json = json.dumps(payload, indent=2, ensure_ascii=False)

    footer = (
        "\n\nReturn JSON ONLY with this exact shape (use the PATH keys shown above):\n"
        "{\n"
        '  "A|C|B": "Cluster Name",\n'
        '  "A|D|A": "Another Name",\n'
        "  ...\n"
        "}\n"
    )

    return header + input_json + footer


def read_json_mapping_interactively(level, expected_keys):
    print(f"\nPaste JSON for level {level}. Type 'EOF' on a new line to finish.")
    print(f"Expected PATH keys: {len(expected_keys)}")
    lines = []
    while True:
        line = input()
        if line.strip() == "EOF":
            break
        lines.append(line)
    txt = "\n".join(lines).strip()
    try:
        mapping = json.loads(txt)
    except json.JSONDecodeError as e:
        print(f"\nInvalid JSON: {e}. Try again.")
        return None

    missing = [k for k in expected_keys if k not in mapping]
    extra = [k for k in mapping.keys() if k not in expected_keys]
    if missing:
        head = ", ".join(missing[:10])
        tail = " ..." if len(missing) > 10 else ""
        print(f"Missing PATH keys: {head}{tail}")
    if extra:
        head = ", ".join(extra[:10])
        tail = " ..." if len(extra) > 10 else ""
        print(f"Ignoring unexpected PATH keys: {head}{tail}")

    mapping = {k: mapping[k] for k in expected_keys if k in mapping}
    return mapping


def build_cluster_id_to_name_mapping(cluster_representatives, path_to_name, cluster_id_to_path, level):
    cluster_id_to_name = {}
    for cluster_id, representatives in cluster_representatives.items():
        path_key = cluster_id_to_path[level][cluster_id]
        if representatives is None:
            cluster_id_to_name[cluster_id] = None
        elif path_key in path_to_name:
            cluster_id_to_name[cluster_id] = path_to_name[path_key]
        else:
            cluster_id_to_name[cluster_id] = None
    return cluster_id_to_name


def create_named_labels(labels_per_level, cluster_id_to_name_per_level, cluster_members, node_info):
    labels_per_level_named = {}
    for level in sorted(labels_per_level.keys()):
        level_names = {}
        cluster_id_to_name = cluster_id_to_name_per_level[level]
        for node_idx in node_info.keys():
            subreddit_name = node_info[node_idx][0]
            if node_idx in labels_per_level[level]:
                cluster_id = labels_per_level[level][node_idx]
                if cluster_id in cluster_id_to_name and cluster_id_to_name[cluster_id] is not None:
                    cluster_name = cluster_id_to_name[cluster_id]
                else:
                    members = cluster_members[level].get(cluster_id, [])
                    if len(members) == 1:
                        cluster_name = subreddit_name
                    else:
                        cluster_name = subreddit_name
            else:
                cluster_name = subreddit_name
            level_names[node_idx] = cluster_name
        labels_per_level_named[level] = level_names
    return labels_per_level_named


def create_named_csv(labels_per_level_named, node_info):
    rows = []
    for node_idx, (subreddit_name, out_degree) in node_info.items():
        if out_degree < MIN_EDGE_COUNT:
            continue
        row = {
            "subreddit_name": subreddit_name,
            "num_outgoing_edges": out_degree
        }
        for level in sorted(labels_per_level_named.keys()):
            if node_idx in labels_per_level_named[level]:
                cluster_name = labels_per_level_named[level][node_idx]
            else:
                cluster_name = subreddit_name
            row[f"level_{level}_cluster"] = cluster_name
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main():
    if not os.path.exists(PKL_IN):
        print(f"Input file not found: {PKL_IN}")
        return

    G, labels_per_level, subreddit_names, node_info = load_all_data()
    cluster_members = build_cluster_membership(labels_per_level, node_info)

    cluster_id_to_name_per_level = {}

    for level in sorted(labels_per_level.keys()):
        cluster_representatives = {}
        cluster_id_to_path = {level: {}}

        for cluster_id, members in cluster_members[level].items():
            representatives = get_cluster_representatives(members)
            cluster_representatives[cluster_id] = representatives
            cluster_id_to_path[level][cluster_id] = cluster_id_to_path_key(cluster_id, level)

        clusters_to_name = [cid for cid, reps in cluster_representatives.items() if reps is not None]
        total = len(cluster_members[level])
        singletons = total - len(clusters_to_name)
        print(f"level {level}: {total} clusters; {len(clusters_to_name)} to name; {singletons} singletons")

        if not clusters_to_name:
            cluster_id_to_name_per_level[level] = {}
            continue

        prompt_text = make_prompt(level, cluster_representatives, cluster_id_to_path)
        if prompt_text is None:
            cluster_id_to_name_per_level[level] = {}
            continue

        prompt_path = os.path.join(PROMPTS_DIR, f"prompt_level_{level}_cluster.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        print(f"prompt written: {prompt_path}")

        expected_keys = [cluster_id_to_path[level][cid] for cid in clusters_to_name]
        while True:
            path_to_name = read_json_mapping_interactively(level, set(expected_keys))
            if path_to_name is not None:
                break

        cluster_id_to_name = build_cluster_id_to_name_mapping(
            cluster_representatives, path_to_name, cluster_id_to_path, level
        )
        cluster_id_to_name_per_level[level] = cluster_id_to_name
        print(f"applied names for level {level}")

    labels_per_level_named = create_named_labels(
        labels_per_level, cluster_id_to_name_per_level, cluster_members, node_info
    )
    with open(PKL_OUT, "wb") as f:
        pickle.dump(labels_per_level_named, f)
    print(f"saved named labels: {PKL_OUT}")

    df = create_named_csv(labels_per_level_named, node_info)
    df.to_csv(CSV_OUT, index=False)
    print(f"saved csv ({len(df)} rows): {CSV_OUT}")
    print("done.")

if __name__ == "__main__":
    main()
