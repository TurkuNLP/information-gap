import pickle
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import sys
import io
import random
import os

def yield_embeddings(pkl_files, args, label_field=None):
    for pkl_file in pkl_files:
        if args.preload_pkl_file_to_memory:
            #print(f"Preloading {pkl_file} to memory", file=sys.stderr, flush=True)
            f=open(pkl_file, "rb")
            emb_pkl=f.read()
            f.close()
            f_emb = io.BytesIO(emb_pkl)
            #print(f"Done. Preloaded {pkl_file} to memory", file=sys.stderr, flush=True)
        else:
            f_emb = open(pkl_file, "rb")
        meta_file_name = pkl_file.replace(".embeddings.pkl", ".examples.jsonl")    
        with open(meta_file_name, "rt") as f_meta:
            while True:
                try:
                    embedding = pickle.load(f_emb)
                    metadata = json.loads(f_meta.readline())
                    if label_field is not None:
                        metadata["label"] = label_field
                    metadata["origin"]=os.path.basename(meta_file_name)
                except EOFError:
                    break
                if np.random.random() < args.global_sample_percentage:
                    yield {"metadata": metadata, "embedding": embedding}


def build_kmeans_model(embeddings_ndarray, num_clusters, args):
    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    print(f"Fitting KMeans with {num_clusters} clusters on {len(embeddings_ndarray)} vectors. Shape: {embeddings_ndarray.shape}", flush=True)
    kmeans.fit(embeddings_ndarray)
    return kmeans



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build and save a KMeans model from one or more embedding .pkl files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--embeddings",
        action="append",
        type=str,
        nargs="+",
        required=True,
        metavar=("LABEL","EMBEDDING_FILES"),
        help="Gather embeddings: LABEL followed by as many .pkl files as you have. You can repeate --embeddings many times for different parts of the dataset."
    )
    parser.add_argument(
        "--pca",
        type=float,
        default=0.0,
        help=f"PCA reduction factor. If 0, no PCA will be performed. Percentage of variance retained."
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path prefix where the following will be saved: .kmeans_model.pkl, .labels.json, .centroids.npy"
    )
    parser.add_argument(
        "--num-clusters",
        type=float,
        default=100,
        help=f"Number of clusters for KMeans. If a float <= 1.0, it will be interpreted as a percentage of the number of embeddings."
    )
    parser.add_argument(
        "--global-sample-percentage",
        type=float,
        default=1.0,
        help=f"Percentage of embeddings to sample when loading the embeddings, the rest will be forever lost."
    )
    parser.add_argument(
        "--fitting-sample-percentage",
        type=float,
        default=1.0,
        help=f"Percentage of embeddings to sample when fitting the PCA+KMeans model, the rest will only be clustered but not used for fitting."
    )
    parser.add_argument(
        "--no-preload-pkl-file-to-memory",
        action="store_false",
        default=True,
        dest="preload_pkl_file_to_memory",
        help=f"Preload the .pkl files to memory."
    )
    args = parser.parse_args()

    embeddings_ndarray = []
    metadata_list = []

    for label_embeddings_files in args.embeddings:
        label, embeddings_files = label_embeddings_files[0], label_embeddings_files[1:]
        print(f"Processing {label} with {len(embeddings_files)} embeddings files: {embeddings_files}", file=sys.stderr, flush=True)
        #this yields a dictionary with "metadata" and "embedding" at a time
        embeddings_generator=yield_embeddings(embeddings_files, label_field=label, args=args)
        for embedding_dict in embeddings_generator:
            embeddings_ndarray.append(embedding_dict["embedding"])
            metadata_list.append(embedding_dict["metadata"])
    embeddings_ndarray = np.vstack(embeddings_ndarray)
    metadata_list = np.array(metadata_list)
    assert len(embeddings_ndarray) == len(metadata_list), "Number of embeddings and labels must match"

    # Shuffle embeddings_ndarray and metadata_list together to maintain correspondence
    permutation = np.random.permutation(len(embeddings_ndarray))
    embeddings_ndarray = embeddings_ndarray[permutation]
    metadata_list = metadata_list[permutation]

    fitting_embeddings_ndarray = embeddings_ndarray[:int(len(embeddings_ndarray) * args.fitting_sample_percentage)]
    if args.pca > 0:
        pca = PCA(n_components=args.pca)
        pca.fit(fitting_embeddings_ndarray)
        fitting_embeddings_ndarray = pca.transform(fitting_embeddings_ndarray)
        embeddings_ndarray = pca.transform(embeddings_ndarray)
        with open(f"{args.output_prefix}.pca_model.pkl", "wb") as f:
            pickle.dump(pca, f)
        print("PCA model saved to", f"{args.output_prefix}.pca_model.pkl", file=sys.stderr, flush=True)
        print(f"PCA: {pca.explained_variance_ratio_.sum():.6f} variance retained in {pca.n_components_} components", file=sys.stderr, flush=True)

    unique_labels = list(set(item["label"] for item in metadata_list))
    print("Loaded", len(embeddings_ndarray), "embeddings for unique labels:", unique_labels,\
        "with shape", embeddings_ndarray.shape, file=sys.stderr, flush=True)

    if args.num_clusters <= 1.0:
        n_clusters = int(args.num_clusters * len(embeddings_ndarray))
    else:
        n_clusters = int(args.num_clusters)
    
    kmeans = build_kmeans_model(fitting_embeddings_ndarray, n_clusters, args)
    cluster_assignments = kmeans.predict(embeddings_ndarray)
 
    with open(f"{args.output_prefix}.kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    with open(f"{args.output_prefix}.metadata.jsonl", "wt") as f:
        for label, cluster_index in zip(metadata_list, cluster_assignments):
            json.dump({"metadata": label, "cluster_index": int(cluster_index)}, f)
            f.write("\n")
    
    with open(f"{args.output_prefix}.centroids.npy", "wb") as f:
        np.save(f, kmeans.cluster_centers_)

    #embeddings_tensor, labels = accumulate_embeddings(args.embeddings_files, args.examples_files)
    #embeddings_tensor, labels = shuffle_and_sample(embeddings_tensor, labels, 0.1)
    #kmeans = build_kmeans_model(embeddings_tensor, args.num_clusters)
    #print(kmeans)
    