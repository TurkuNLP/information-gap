import json
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from collections import Counter, defaultdict

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

def read_documents(document_files):
    documents = []
    for document_file in document_files:
        with open(document_file, 'r') as f:
            for line in f:
                documents.append(json.loads(line))
    return documents


def get_cluster_labels(documents):
    # get list of labels for each cluster
    cluster_labels = {}
    for doc in documents:
        cid = doc["cluster_index"]
        label = doc["metadata"]["label"]
        if cid not in cluster_labels:
            cluster_labels[cid] = []
        cluster_labels[cid].append(label)
    return cluster_labels


def cluster_historgram(documents, args):

    clusteridx2labels = get_cluster_labels(documents)
    
    # Labels distribution for each cluster: native / (native + translated)
    cluster_distrib = [clusteridx2labels[i].count(args.native_label) / (clusteridx2labels[i].count(args.native_label) + clusteridx2labels[i].count(args.translated_label)) for i in range(len(clusteridx2labels))]
    #print([(i,v) for i,v in enumerate(cluster_distrib)], file=sys.stderr)


    # order clusters by percentage of native documents (indices in ascending order of cluster_distrib)
    order_by_distrib = np.argsort(cluster_distrib)

    unique_labels = set(l for labels in clusteridx2labels.values() for l in labels)
    labels = [args.native_label, args.translated_label] + [l for l in unique_labels if l != args.native_label and l != args.translated_label]

    # make stacked bars for each label separately
    x_items = order_by_distrib # cluster idx order
    x_pos = np.arange(len(x_items)) # cluster position
    width = 0.8
    bottom = np.zeros(len(x_items))
    label_to_color = {label: COLOR_PALETTE[i] for i, label in enumerate(labels)} # define colors
    bar_containers = []
    for label in labels: # make bars for each label separately
        counts = np.array([clusteridx2labels[cid].count(label) for cid in x_items])
        bars = plt.bar(x_pos, counts, width, bottom=bottom, label=label, color=label_to_color[label])
        bar_containers.append(bars)
        bottom += counts

    # add annotation
    try:
        import mplcursors
        c = mplcursors.cursor(bar_containers, hover=mplcursors.HoverMode.Transient, highlight=False)
        @c.connect("add")
        def on_add(sel):
            cluster_idx = x_items[sel.index]
            native = cluster_distrib[cluster_idx] 
            sel.annotation.set(text=f"Cluster {x_items[sel.index]}\nnative: {native:.1%}\nsize: {len(clusteridx2labels[cluster_idx])}\nlabels:{Counter(clusteridx2labels[cluster_idx]).most_common(10)}")
            sel.annotation.get_bbox_patch().set(alpha=1.0)  # fully opaque hover box
    except ImportError:
        print("mplcursors not installed, skipping hover tooltips", file=sys.stderr)

    plt.xlabel("Clusters")
    plt.ylabel("Number of documents")
    plt.tight_layout()
    plt.show()



def plot_centroids_scatter(centroids, documents, args):
    # plot cluster centroids, numpy array of shape (n_clusters, embedding_dim) with t-SNE
    # color each centroid by cluster's pre-training data distribution (native vs translated)

    clusteridx2labels = get_cluster_labels(documents)

    # Labels distribution for each cluster
    cluster_distrib = [clusteridx2labels[i].count(args.native_label) / (clusteridx2labels[i].count(args.native_label) + clusteridx2labels[i].count(args.translated_label)) for i in range(centroids.shape[0])]
    print([(i,v) for i,v in enumerate(cluster_distrib)], file=sys.stderr)

    # t-SNE projection (use random_state for reproducibility)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(centroids)

    # make scatter plot
    cmap = plt.get_cmap("RdYlGn")
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_distrib, cmap=cmap, s=60, edgecolor='k')

    # Use mplcursors to add hover annotations if available
    try:
        import mplcursors
        c = mplcursors.cursor(scatter, hover=mplcursors.HoverMode.Transient, highlight=False)
        @c.connect("add")
        def on_add(sel):
            sel.annotation.set(text=f"Cluster {sel.index}\nnative: {cluster_distrib[sel.index]:.1%}\nsize: {len(clusteridx2labels[sel.index])}")
            sel.annotation.get_bbox_patch().set(alpha=1.0)
    except ImportError:
        print("mplcursors not installed, skipping hover tooltips", file=sys.stderr)

    cbar = plt.colorbar(scatter)
    cbar.set_label('% native documents', rotation=270, labelpad=20)
    plt.title('Cluster Centroids t-SNE Projection\nColor: native (green) → translated (red) in pre-training data (%)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()


def plot_centroids_density(centroids, documents, benchmarks, args):
    
    clusteridx2labels = get_cluster_labels(documents)

    # Labels distribution for each cluster
    cluster_distrib = [clusteridx2labels[i].count(args.native_label) / (clusteridx2labels[i].count(args.native_label) + clusteridx2labels[i].count(args.translated_label)) for i in range(centroids.shape[0])]
    print([(i,v) for i,v in enumerate(cluster_distrib)], file=sys.stderr)

    # t-SNE projection (use random_state for reproducibility)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(centroids)
    print(embedding.shape, file=sys.stderr)

    # make scatter plot
    cmap = plt.get_cmap("RdYlGn")
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_distrib, cmap=cmap, s=80, edgecolor="none", alpha=0.5)

    # add benchmarks as black dots
    # benchmark labels (cluster indices) must be connected to tsne embedding by cluster_index
    benchmark_points = []
    for example in benchmarks:
        cidx = example["cluster_index"]
        benchmark_points.append(embedding[cidx])

    label_counter = {k:Counter(v) for k,v in get_cluster_labels(documents+benchmarks).items()}
    print(label_counter, file=sys.stderr)

    benchmark_points = np.array(benchmark_points)
    bscatter =plt.scatter(benchmark_points[:, 0], benchmark_points[:, 1], c="black", marker="x", s=40, edgecolor="none", alpha=1)

    # Use mplcursors to add hover annotations if available, only for benchmarkpoints
    try:
        import mplcursors
        cluster_indices = [example["cluster_index"] for example in benchmarks]
        c = mplcursors.cursor(bscatter, hover=mplcursors.HoverMode.Transient, highlight=False)
        @c.connect("add")
        def on_add(sel):
            cluster_idx = cluster_indices[sel.index]
            labels_info = label_counter.get(cluster_idx, {})
            sel.annotation.set(text=f"Cluster {cluster_idx}\nLabels: {labels_info.most_common(10) if hasattr(labels_info, 'most_common') else labels_info}")
            sel.annotation.get_bbox_patch().set(alpha=1.0)
    except ImportError:
        print("mplcursors not installed, skipping hover tooltips", file=sys.stderr)

    cbar = plt.colorbar(scatter)
    cbar.set_label('% native documents', rotation=270, labelpad=20)
    plt.title('Cluster Centroids t-SNE Projection\nColor: native (green) → translated (red) in pre-training data (%)\nBlack: clusters with benchmark data')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()


def main(args):
    
    #read data
    cluster_centroids = np.load(args.cluster_centroids)
    print("Cluster centroids shape:", cluster_centroids.shape, file=sys.stderr)
    documents = read_documents(args.pre_training_documents)
    print("Pre-training documents:", len(documents), file=sys.stderr)
    benchmarks = read_documents(args.benchmarks)
    print("Benchmark documents:", len(benchmarks), file=sys.stderr)


    # plot centroids scatter for documents (native and translated)
    plot_centroids_scatter(cluster_centroids, documents, args)

    # plot centroids density with benchmarks
    plot_centroids_density(cluster_centroids, documents, benchmarks, args)

    # plot cluster histogram for documents (native and translated) and benchmarks
    cluster_historgram(documents+benchmarks, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster-centroids', type=str, default='toy_data/toy_cluster_data.npy')
    parser.add_argument('--pre-training-documents', type=str, nargs='+', default=['toy_data/toy_documents.jsonl'])
    parser.add_argument('--benchmarks', type=str, nargs='+', default=[])
    parser.add_argument('--native-label', type=str, default="native", help="Label for native documents")
    parser.add_argument('--translated-label', type=str, default="tropus", help="Label for translated documents")
    parser.add_argument('--benchmark-label', type=str, default="benchmark", help="Label for benchmark documents")
    args = parser.parse_args()
    main(args)

