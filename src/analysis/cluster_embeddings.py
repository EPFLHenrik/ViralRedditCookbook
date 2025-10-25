import os
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.utils.data_utils import load_reddit_graph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'clustering_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DEPTH = 4
PCA_VARIANCE_THRESHOLD = 0.99
PCA_MAX_COMPONENTS = 200
K_NEIGHBORS = 15
MIN_COMMUNITY_SIZE = 5
MIN_SPLIT_SIZE = 20
N_EXEMPLARS = 5

print("Loading Reddit graph...")
G = load_reddit_graph()
print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def get_active_subreddits(graph, min_out_degree=1000):
    out_degrees = dict(graph.out_degree())
    active_nodes = [node for node, degree in out_degrees.items() if degree > min_out_degree]
    actual_count = len(active_nodes)
    min_degree = min(out_degrees[node] for node in active_nodes) if active_nodes else 0
    max_degree = max(out_degrees[node] for node in active_nodes) if active_nodes else 0
    print(f"Selected {actual_count} subreddits with >{min_out_degree} outgoing edges")
    print(f"  Out-degree range: {min_degree} to {max_degree} posts")
    return set(active_nodes)


def load_and_filter_embeddings(embeddings_path, valid_subreddits):
    print(f"Loading embeddings from {embeddings_path}...")
    df = pd.read_csv(embeddings_path, header=None)
    subreddit_names = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype(np.float32)
    print(f"Loaded {len(subreddit_names)} subreddits with {embeddings.shape[1]}-dimensional embeddings")
    valid_set = set(valid_subreddits)
    mask = np.array([name in valid_set for name in subreddit_names])
    filtered_names = subreddit_names[mask]
    filtered_embeddings = embeddings[mask]
    print(f"Filtered to {len(filtered_names)} subreddits present in graph")
    return filtered_names, filtered_embeddings


def preprocess_embeddings(embeddings):
    print("Preprocessing embeddings...")
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
    n_samples, n_features = embeddings_normalized.shape
    max_components = min(PCA_MAX_COMPONENTS, n_samples, n_features)
    pca = PCA(n_components=max_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_normalized)
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum_variance, PCA_VARIANCE_THRESHOLD) + 1
    n_components = min(n_components, max_components)
    embeddings_reduced = embeddings_pca[:, :n_components]
    total_variance = cumsum_variance[n_components - 1]
    print(f"PCA reduced to {n_components} components (variance retained: {total_variance:.4f})")
    embeddings_final = normalize(embeddings_reduced, norm='l2', axis=1)
    return embeddings_final, pca


def build_knn_graph(embeddings, k=K_NEIGHBORS, prune_threshold=0.3):
    print(f"Building k-NN graph with k={k}...")
    n_samples = embeddings.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='auto', n_jobs=-1)
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    similarities = 1 - distances
    edge_weights = {}
    for i in range(n_samples):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            sim = similarities[i, j_idx]
            edge = tuple(sorted([i, j]))
            if edge in edge_weights:
                edge_weights[edge] = max(edge_weights[edge], sim)
            else:
                edge_weights[edge] = sim
    edges_pruned = [(i, j, w) for (i, j), w in edge_weights.items() if w >= prune_threshold]
    print(f"Built k-NN graph: {n_samples} nodes, {len(edges_pruned)} edges (after pruning < {prune_threshold})")
    G_sim = nx.Graph()
    G_sim.add_nodes_from(range(n_samples))
    for i, j, w in edges_pruned:
        G_sim.add_edge(i, j, weight=w)
    return G_sim


def nx_to_igraph(G_nx, node_mapping=None):
    if node_mapping is None:
        node_mapping = {node: i for i, node in enumerate(G_nx.nodes())}
    edges = [(node_mapping[u], node_mapping[v]) for u, v in G_nx.edges()]
    weights = [G_nx[u][v].get('weight', 1.0) for u, v in G_nx.edges()]
    n_nodes = len(node_mapping)
    G_ig = ig.Graph(n=n_nodes, edges=edges, directed=False)
    G_ig.es['weight'] = weights
    return G_ig, node_mapping


def compute_modularity(G_ig, membership):
    if len(set(membership)) <= 1:
        return 0.0
    return G_ig.modularity(membership, weights='weight')


def leiden_clustering(G_nx, resolution=1.0):
    if G_nx.number_of_nodes() == 0:
        return []
    G_ig, node_mapping = nx_to_igraph(G_nx)
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    partition = G_ig.community_leiden(
        objective_function='modularity',
        weights='weight',
        resolution=resolution,
        n_iterations=-1
    )
    membership = partition.membership
    result = {}
    for ig_id, cluster_id in enumerate(membership):
        nx_id = reverse_mapping[ig_id]
        result[nx_id] = cluster_id
    return result


def should_split_community(G_nx, parent_size, child_labels, parent_modularity):
    if parent_size < MIN_SPLIT_SIZE:
        return False
    n_children = len(set(child_labels.values()))
    if n_children < 2:
        return False
    if n_children > parent_size / MIN_COMMUNITY_SIZE:
        return False
    child_sizes = Counter(child_labels.values())
    if any(size < MIN_COMMUNITY_SIZE for size in child_sizes.values()):
        return False
    largest_child = max(child_sizes.values())
    if largest_child > 0.75 * parent_size:
        return False
    return True


def recursive_leiden_refinement(
    G_sim,
    subreddit_names,
    embeddings,
    parent_indices=None,
    parent_id="0",
    depth=0,
    resolution=1.0
):
    if parent_indices is None:
        parent_indices = list(range(len(subreddit_names)))
    hierarchy = {}
    labels_per_level = {depth: {}}
    exemplars = {}
    subgraph = G_sim.subgraph(parent_indices).copy()
    print(f"{'  ' * depth}Level {depth}, ID {parent_id}: {len(parent_indices)} nodes")
    for idx in parent_indices:
        labels_per_level[depth][idx] = parent_id
    parent_indices_array = np.array(parent_indices)
    parent_embeddings = embeddings[parent_indices_array]
    centroid = parent_embeddings.mean(axis=0)
    distances_to_centroid = np.linalg.norm(parent_embeddings - centroid, axis=1)
    exemplar_indices = parent_indices_array[np.argsort(distances_to_centroid)[:N_EXEMPLARS]]
    exemplars[parent_id] = exemplar_indices.tolist()
    if depth >= MAX_DEPTH or len(parent_indices) < MIN_SPLIT_SIZE:
        return hierarchy, labels_per_level, exemplars
    local_labels = leiden_clustering(subgraph, resolution=resolution * (1.2 ** depth))
    global_labels = {idx: local_labels.get(idx, 0) for idx in parent_indices}
    G_ig_parent, _ = nx_to_igraph(subgraph)
    parent_membership = [0] * len(parent_indices)
    parent_modularity = 0.0
    if should_split_community(subgraph, len(parent_indices), local_labels, parent_modularity):
        child_communities = {}
        for idx in parent_indices:
            child_label = global_labels[idx]
            if child_label not in child_communities:
                child_communities[child_label] = []
            child_communities[child_label].append(idx)
        print(f"{'  ' * depth}  → Split into {len(child_communities)} children")
        for child_idx, child_indices in enumerate(child_communities.values()):
            child_id = f"{parent_id}.{child_idx}"
            hierarchy[child_id] = parent_id
            child_hierarchy, child_labels, child_exemplars = recursive_leiden_refinement(
                G_sim,
                subreddit_names,
                embeddings,
                parent_indices=child_indices,
                parent_id=child_id,
                depth=depth + 1,
                resolution=resolution
            )
            hierarchy.update(child_hierarchy)
            for level, level_labels in child_labels.items():
                if level not in labels_per_level:
                    labels_per_level[level] = {}
                labels_per_level[level].update(level_labels)
            exemplars.update(child_exemplars)
    else:
        print(f"{'  ' * depth}  → No split (quality criteria not met)")
    return hierarchy, labels_per_level, exemplars


def compute_umap_embedding(embeddings):
    print("Computing UMAP embedding...")
    umap_model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        n_jobs=-1
    )
    umap_coords = umap_model.fit_transform(embeddings)
    print(f"UMAP embedding computed: {umap_coords.shape}")
    return umap_coords


def save_results(
    subreddit_names,
    embeddings,
    hierarchy,
    labels_per_level,
    exemplars,
    umap_coords
):
    print("Saving results...")
    hierarchy_path = os.path.join(RESULTS_DIR, 'hierarchy.json')
    with open(hierarchy_path, 'w') as f:
        json.dump(hierarchy, f, indent=2)
    print(f"  Saved hierarchy to {hierarchy_path}")
    labels_path = os.path.join(RESULTS_DIR, 'labels_per_level.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(labels_per_level, f)
    print(f"  Saved labels to {labels_path}")
    exemplars_with_names = {}
    for cluster_id, indices in exemplars.items():
        exemplars_with_names[cluster_id] = [
            {'index': int(idx), 'subreddit': subreddit_names[idx]}
            for idx in indices
        ]
    exemplars_path = os.path.join(RESULTS_DIR, 'exemplars.json')
    with open(exemplars_path, 'w') as f:
        json.dump(exemplars_with_names, f, indent=2)
    print(f"  Saved exemplars to {exemplars_path}")
    umap_data = {
        'subreddit_names': subreddit_names.tolist(),
        'umap_x': umap_coords[:, 0].tolist(),
        'umap_y': umap_coords[:, 1].tolist(),
        'labels_level_0': [labels_per_level[0].get(i, None) for i in range(len(subreddit_names))]
    }
    umap_path = os.path.join(RESULTS_DIR, 'umap_embedding.json')
    with open(umap_path, 'w') as f:
        json.dump(umap_data, f)
    print(f"  Saved UMAP data to {umap_path}")
    embeddings_path = os.path.join(RESULTS_DIR, 'processed_embeddings.npz')
    np.savez(
        embeddings_path,
        subreddit_names=subreddit_names,
        embeddings=embeddings,
        umap_coords=umap_coords
    )
    print(f"  Saved processed embeddings to {embeddings_path}")
    print("\n" + "=" * 70)
    print("CLUSTERING SUMMARY")
    print("=" * 70)
    print(f"Total subreddits: {len(subreddit_names)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Number of levels: {len(labels_per_level)}")
    print(f"Number of clusters: {len(exemplars)}")
    for level in sorted(labels_per_level.keys()):
        level_labels = labels_per_level[level]
        n_clusters = len(set(level_labels.values()))
        print(f"  Level {level}: {n_clusters} clusters")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING PIPELINE")
    print("=" * 70 + "\n")
    all_subreddits = set(G.nodes())
    print(f"Using all {len(all_subreddits)} subreddits from the graph")
    embeddings_path = os.path.join(DATA_DIR, 'web-redditEmbeddings-subreddits.csv')
    subreddit_names, embeddings_raw = load_and_filter_embeddings(embeddings_path, all_subreddits)
    embeddings_processed, pca_model = preprocess_embeddings(embeddings_raw)
    G_sim = build_knn_graph(embeddings_processed, k=K_NEIGHBORS)
    print("\nRunning hierarchical Leiden clustering...")
    hierarchy, labels_per_level, exemplars = recursive_leiden_refinement(
        G_sim,
        subreddit_names,
        embeddings_processed
    )
    umap_coords = compute_umap_embedding(embeddings_processed)
    save_results(
        subreddit_names,
        embeddings_processed,
        hierarchy,
        labels_per_level,
        exemplars,
        umap_coords
    )
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
