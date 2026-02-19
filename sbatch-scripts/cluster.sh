#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=cluster
#SBATCH --output=logs/cluster-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=small
#SBATCH --cpus-per-task=1
#SBATCH --cores 8
#SBATCH --mem=40GB
#SBATCH --account=project_2000539

module load pytorch

mkdir -p logs

embpath="/scratch/project_2000539/information-gap-data/samples/embeddings-e5"
clusterpath="/scratch/project_2000539/information-gap-data/clusters"

for tmodel in opus tower9b tower72b; do
    echo $tmodel
    echo "$clusterpath/e5-native-$tmodel-10K"
    mkdir -p $clusterpath/e5-native-$tmodel-10K
    python3 ../gen_kmeans.py --embeddings native $embpath/native-fin-10K-sample.0.embeddings.pkl --embeddings $tmodel $embpath/translated-$tmodel-fin-10K-sample.0.embeddings.pkl --global-sample 1.0 --num-clusters 0.1 --fitting 1.0 --pca 0.9 --output-prefix $clusterpath/e5-native-$tmodel-10K/clusters
    
    #python3 ../gen_kmeans.py --embeddings native $embpath/native-fin-1M-sample.0.embeddings.pkl --embeddings $tmodel $embpath/translated-$tmodel-fin-1M-sample.0.embeddings.pkl --global-sample 1.0 --num-clusters 0.1 --fitting 0.1 --pca 0.9 --output-prefix $clusterpath/e5-native-$tmodel-1M/clusters
done
