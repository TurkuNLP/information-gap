#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=cluster-benchmarks
#SBATCH --output=logs/cluster-benchmarks-%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=small
#SBATCH --cpus-per-task=1
#SBATCH --cores 8
#SBATCH --mem=20GB
#SBATCH --account=project_2000539

module load pytorch

mkdir -p logs

embpath="/scratch/project_2000539/information-gap-data/benchmarks/embeddings-e5"
clusterpath="/scratch/project_2000539/information-gap-data/clusters"



for ds in $embpath/TurkuNLP-finbenchv2-*.pkl; do
    for cmodel in $clusterpath/*; do

        echo $ds
        echo $cmodel

        dsname=$(basename $ds)
        dsprefix=${dsname%.embeddings.pkl}
        echo $dsprefix

        python3 ../get_clusters_for_new_embeddings.py --embeddings $dsprefix $ds --model-prefix $cmodel/clusters --output-prefix $cmodel/$dsprefix
        echo "done"
    done
done
