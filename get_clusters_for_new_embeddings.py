import argparse
import json
import sys
import gen_kmeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute cluster IDs for new embeddings using saved PCA+KMeans models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input_grp = parser.add_argument_group("input/output")
    input_grp.add_argument(
        "--embeddings",
        action="append",
        type=str,
        nargs="+",
        required=True,
        metavar=("LABEL","EMBEDDING_FILES"),
        help="Gather embeddings: LABEL followed by as many .pkl files as you have. You can repeat --embeddings many times."
    )
    input_grp.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="Path prefix where the saved models live (.pca_model.pkl, .kmeans_model.pkl)"
    )
    input_grp.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path prefix for output .metadata.jsonl"
    )
    input_grp.add_argument(
        "--global-sample-percentage",
        type=float,
        default=1.0,
        help="Percentage of embeddings to sample when loading, the rest will be forever lost."
    )
    input_grp.add_argument(
        "--no-preload-pkl-file-to-memory",
        action="store_false",
        default=True,
        dest="preload_pkl_file_to_memory",
        help="Preload the .pkl files to memory."
    )
    args = parser.parse_args()

    embeddings_ndarray, metadata_list = gen_kmeans.load_embeddings(args.embeddings, args)
    unique_labels = list(set(item["label"] for item in metadata_list))
    print("Loaded", len(embeddings_ndarray), "embeddings for unique labels:", unique_labels,
        "with shape", embeddings_ndarray.shape, file=sys.stderr, flush=True)

    pca, kmeans = gen_kmeans.load_models(args.model_prefix)
    if pca is not None:
        embeddings_ndarray = pca.transform(embeddings_ndarray)
    cluster_assignments = kmeans.predict(embeddings_ndarray)

    with open(f"{args.output_prefix}.metadata.jsonl", "wt") as f:
        for meta, cluster_index in zip(metadata_list, cluster_assignments):
            json.dump({"metadata": meta, "cluster_index": int(cluster_index)}, f)
            f.write("\n")
