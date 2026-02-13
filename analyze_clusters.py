import json
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from collections import Counter, defaultdict

#COLOR_PALETTE = ["#7eb6d9", "#f4a582", "#92c5de", "#d4a5a5", "#a8d4a8", "#d4c4e0", "#f5e6a3", "#b8e0d2", "#e8c9e8", "#ffd9b3"]
# Green and red first (main pair), then extra pastels for plots
COLOR_PALETTE = ["#5da86a", "#d65c5c", "#92c5de", "#d4a5a5", "#7eb6d9", "#d4c4e0", "#f5e6a3", "#b8e0d2", "#e8c9e8", "#ffd9b3"]

# //////////////////////////////////////////
# produce toy cluster centroids: shape (100, 1024)
toy_cluster_data = np.random.rand(100, 1024)
np.save('toy_data/toy_cluster_data.npy', toy_cluster_data)

# produce toy documents (1000 documents)
with open('toy_data/toy_documents.jsonl', 'w') as f:
    for i in range(1000):
        document = {
            "cluster_index": np.random.randint(0, 100),
            "label": "native" if np.random.rand() < 0.5 else "translated"
        }
        print(json.dumps(document), file=f)
# ///////////////////////////////////////////

def read_documents(args):
    documents = []
    for document_file in args.documents:
        with open(document_file, 'r') as f:
            for line in f:
                documents.append(json.loads(line))
    return documents



def cluster_historgram(documents, max_clusters=100):
    # get list of labels for each cluster
    cluster_labels = {}
    for doc in documents:
        cid = doc["cluster_index"]
        label = doc["metadata"]["label"]
        if cid not in cluster_labels:
            cluster_labels[cid] = []
        cluster_labels[cid].append(label)
    unique_labels = set(l for labels in cluster_labels.values() for l in labels)
    # ordered liast of labels
    if "native" in unique_labels:
        label_order = ["native"] + [l for l in unique_labels if l != "native"]
    else:
        label_order = list(unique_labels)

    # calculate percentage of native documents for each cluster
    cluster_pct_native = {cid: labels.count("native") / len(labels) for cid, labels in cluster_labels.items()}
    # order clusters by percentage of native documents
    ordered_clusters = [k for k, v in sorted(cluster_pct_native.items(), key=lambda item: item[1])]

    # Bar order = order of items in x; use positions 0,1,2,... so bars follow list order
    x_items = ordered_clusters
    x_pos = np.arange(len(x_items))
    width = 0.8
    bottom = np.zeros(len(x_items))

    label_to_color = {label: COLOR_PALETTE[i] for i, label in enumerate(label_order)} # define colors
    bar_containers = []
    for label in label_order: # make bars for each label separately
        counts = np.array([cluster_labels[cid].count(label) for cid in x_items])
        bars = plt.bar(x_pos, counts, width, bottom=bottom, label=label, color=label_to_color[label])
        bar_containers.append(bars)
        bottom += counts

    try:
        import mplcursors
        c = mplcursors.cursor(bar_containers, hover=mplcursors.HoverMode.Transient, highlight=False)
        @c.connect("add")
        def on_add(sel):
            cluster_id = x_items[sel.index]
            pct_native = cluster_pct_native[cluster_id]
            cluster_size = len(cluster_labels[cluster_id])
            sel.annotation.set(text=f"Cluster {cluster_id}\nnative: {pct_native:.1%}\nsize: {cluster_size}")
            sel.annotation.get_bbox_patch().set(alpha=1.0)  # fully opaque hover box
    except ImportError:
        print("mplcursors not installed, skipping hover tooltips", file=sys.stderr)

    plt.xlabel("Cluster")
    plt.ylabel("Number of documents")
    plt.xticks(x_pos, x_items)
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()



def plot_centroids_scatter(centroids, documents):
    # centroids is numpy array of shape (n_clusters, embedding_dim)
    # documents is list of dictionaries with keys 'cluster_index' and 'label'

    # Determine cluster ID count
    cluster_labels = {i: [] for i in range(centroids.shape[0])}
    for doc in documents:
        cluster_labels[doc["cluster_index"]].append(doc["metadata"]["label"])
    # assert that we do not have empty clusters
    assert all(len(labels) > 0 for labels in cluster_labels.values()), "Empty clusters found"

    # Turn labels into label percentages (assume two labels: 'native' and 'translated')
    cluster_values = [cluster_labels[i].count('native') / len(cluster_labels[i]) for i in range(centroids.shape[0])]
    print([(i,v) for i,v in enumerate(cluster_values)], file=sys.stderr)

    # t-SNE projection (use random_state for reproducibility)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(centroids)

    cmap = plt.get_cmap("RdYlGn")
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_values,
        cmap=cmap,
        s=60,
        edgecolor='k'
    )

    # Show cluster ID only when hovering over the point
    # Use mplcursors to add hover tooltips
    try:
        import mplcursors
        c = mplcursors.cursor(scatter, hover=mplcursors.HoverMode.Transient, highlight=False)
        @c.connect("add")
        def on_add(sel):
            cluster_id = sel.index
            pct_native = cluster_values[cluster_id]
            cluster_size = len(cluster_labels[cluster_id])
            sel.annotation.set(text=f"Cluster {cluster_id}\nnative: {pct_native:.1%}\nsize: {cluster_size}")
            sel.annotation.get_bbox_patch().set(alpha=1.0)  # fully opaque hover box
    except ImportError:
        print("mplcursors not installed, skipping hover tooltips", file=sys.stderr)


    cbar = plt.colorbar(scatter)
    cbar.set_label('% native documents', rotation=270, labelpad=20)
    plt.title('Cluster Centroids t-SNE Projection\nColor: % native (green → red)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()
    pass

def main(args):
    cluster_centroids = np.load(args.cluster_centroids)
    print("Cluster centroids shape:", cluster_centroids.shape, file=sys.stderr)
    documents = read_documents(args)
    print("Documents:", len(documents), file=sys.stderr)
    plot_centroids_scatter(cluster_centroids, documents)
    cluster_historgram(documents, max_clusters=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster-centroids', type=str, default='toy_data/toy_cluster_data.npy')
    parser.add_argument('--documents', type=str, nargs='+', default=['toy_data/toy_documents.jsonl'])
    args = parser.parse_args()
    main(args)

