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
import gzip

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
        meta_file_name = pkl_file.replace(".embeddings.pkl", ".examples.jsonl.gz") # try gz first, if does not exist default to jsonl
        if not os.path.isfile(meta_file_name):
            meta_file_name = pkl_file.replace(".embeddings.pkl", ".examples.jsonl")
        with gzip.open(meta_file_name, "rt") if meta_file_name.endswith(".gz") else open(meta_file_name, "rt") as f_meta:
            while True:
                try:
                    embedding = pickle.load(f_emb)
                    metadata = json.loads(f_meta.readline())
                    if label_field is not None:
                        metadata["label"] = label_field
                    if "origin" not in metadata:
                        metadata["origin"]=os.path.basename(meta_file_name)
                except EOFError:
                    break
                if np.random.random() < args.global_sample_percentage:
                    yield {"metadata": metadata, "embedding": embedding}
        if not args.preload_pkl_file_to_memory:
            f_emb.close()

def load_embeddings(embeddings_files, args):
    embeddings_ndarray = []
    metadata_list = []
    for label_embeddings_files in embeddings_files:
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
    return embeddings_ndarray, metadata_list

def shuffle_embeddings(embeddings_ndarray, metadata_list):
    permutation = np.random.permutation(len(embeddings_ndarray))
    embeddings_ndarray = embeddings_ndarray[permutation]
    metadata_list = metadata_list[permutation]
    return embeddings_ndarray, metadata_list

def load_models(model_prefix):
    """Load pca and kmeans from disk. pca is None if no PCA was used."""
    pca_path = f"{model_prefix}.pca_model.pkl"
    pca = pickle.load(open(pca_path, "rb")) if os.path.exists(pca_path) else None
    kmeans = pickle.load(open(f"{model_prefix}.kmeans_model.pkl", "rb"))
    return pca, kmeans

def fit_models(embeddings_ndarray, args):
    if args.fitting_sample_percentage < 1.0:
        fitting_embeddings_ndarray = embeddings_ndarray[:int(len(embeddings_ndarray) * args.fitting_sample_percentage)]
    else:
        fitting_embeddings_ndarray = embeddings_ndarray
    if args.pca > 0:
        pca = PCA(n_components=args.pca)
        pca.fit(fitting_embeddings_ndarray)
        fitting_embeddings_ndarray = pca.transform(fitting_embeddings_ndarray)
        with open(f"{args.output_prefix}.pca_model.pkl", "wb") as f:
            pickle.dump(pca, f)
        print("PCA model saved to", f"{args.output_prefix}.pca_model.pkl", file=sys.stderr, flush=True)
        print(f"PCA: {pca.explained_variance_ratio_.sum():.6f} variance explained in {pca.n_components_} components", file=sys.stderr, flush=True)
    else:
        pca = None

    if args.num_clusters <= 1.0:
        n_clusters = int(args.num_clusters * len(embeddings_ndarray))
    else:
        n_clusters = int(args.num_clusters)
    print(f"Fitting KMeans with {n_clusters} clusters on vectors of shape: {fitting_embeddings_ndarray.shape}", flush=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(fitting_embeddings_ndarray)
 
    with open(f"{args.output_prefix}.kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    with open(f"{args.output_prefix}.centroids.npy", "wb") as f:
        np.save(f, kmeans.cluster_centers_)

    return pca, kmeans

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build and save a KMeans model from one or more embedding .pkl files",
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
        help="Gather embeddings: LABEL followed by as many .pkl files as you have. You can repeate --embeddings many times for different parts of the dataset."
    )
    input_grp.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path prefix where the following will be saved: .kmeans_model.pkl, .labels.json, .centroids.npy, .pca_model.pkl"
    )
    input_grp.add_argument(
        "--global-sample-percentage",
        type=float,
        default=1.0,
        help=f"Percentage of embeddings to sample when loading the embeddings, the rest will be forever lost."
    )
    input_grp.add_argument(
        "--no-shuffle",
        action="store_false",
        default=True,
        dest="shuffle",
        help=f"Random shuffle the data."
    )
    input_grp.add_argument(
        "--no-preload-pkl-file-to-memory",
        action="store_false",
        default=True,
        dest="preload_pkl_file_to_memory",
        help=f"Preload the .pkl files to memory."
    )

    model_grp = parser.add_argument_group("model fitting")
    model_grp.add_argument(
        "--fitting-sample-percentage",
        type=float,
        default=1.0,
        help=f"Percentage of embeddings to sample when fitting the PCA+KMeans model, the rest will only be clustered but not used for fitting."
    )
    model_grp.add_argument(
        "--pca",
        type=float,
        default=0.0,
        help=f"PCA reduction factor. If 0, no PCA will be performed. Percentage of variance retained."
    )
    model_grp.add_argument(
        "--num-clusters",
        type=float,
        default=100,
        help=f"Number of clusters for KMeans. If a float <= 1.0, it will be interpreted as a percentage of the number of embeddings."
    )
    args = parser.parse_args()

    
    embeddings_ndarray, metadata_list = load_embeddings(args.embeddings, args)
    unique_labels = list(set(item["label"] for item in metadata_list))
    print("Loaded", len(embeddings_ndarray), "embeddings for unique labels:", unique_labels,\
        "with shape", embeddings_ndarray.shape, file=sys.stderr, flush=True)

    if args.shuffle:
        embeddings_ndarray, metadata_list = shuffle_embeddings(embeddings_ndarray, metadata_list)
    
    pca, kmeans = fit_models(embeddings_ndarray, args)
    embeddings_ndarray = pca.transform(embeddings_ndarray)
    cluster_assignments = kmeans.predict(embeddings_ndarray)
    with open(f"{args.output_prefix}.metadata.jsonl", "wt") as f:
        for label, cluster_index in zip(metadata_list, cluster_assignments):
            json.dump({"metadata": label, "cluster_index": int(cluster_index)}, f)
            f.write("\n")

    
