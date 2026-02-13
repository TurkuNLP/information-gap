# information-gap

Code for an information gap experiment. 

**Work in progress, completely unusable by anyone outside.**

# Example to build new clustering

This will produce a number of clusters/e5_20K_cl_2000_pca_0.9.* files with the models and cluster ids and stuff
Note how the different embeddings are given using repeated use of --embeddings LABEL FILES arg

python3 gen_kmeans.py --embeddings native /scratch/project_2000539/information-gap-data/samples/embeddings-e5/native-fin-10K.0.embeddings.pkl --embeddings tropus /scratch/project_2000539/information-gap-data/samples/embeddings-e5/translated-opus-fin-10K.0.embeddings.pkl --global-sample 1.0 --num-clusters 0.1 --fitting 0.5 --output-prefix clusters/e5_20K_cl_2000_pca_0.9 --pca 0.9

# Example to assign ready 

This will assign new embeddings to cluster ids, producing ddelme.metadata.jsonl (note -> uses same old embeddings as the command above, but of course you'd run it with different ones). Same --embeddings logic as above, but now what was --output-prefix becomes --model-prefix since we just load the models. And --output-prefix here does not store any models, simply produces the relevant metadata file

python3 get_clusters_for_new_embeddings.py --embeddings native /scratch/project_2000539/information-gap-data/samples/embeddings-e5/native-fin-10K.0.embeddings.pkl --embeddings tropus /scratch/project_2000539/information-gap-data/samples/embeddings-e5/translated-opus-fin-10K.0.embeddings.pkl --model-prefix clusters/e5_20K_cl_2000_pca_0.9 --output-prefix ddelme