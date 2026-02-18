#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=embed-benchmarks
#SBATCH --output=logs/embed-benchmarks-%j.out
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --cores 4
#SBATCH --mem=20GB
#SBATCH --account=project_2000539

module load pytorch

mkdir -p logs

export HF_HOME="/scratch/project_2000539/jenna/hf-cache"
export HF_DATASETS_CACHE="/scratch/project_2000539/jenna/hf-cache"


code="/scratch/project_2000539/jenna/git_checkout/embedding-prompt-analysis"

echo "$code/compute_embeddings.py"




for ds in ./benchmarks/TurkuNLP-finbenchv2-*.jsonl; do

    echo $ds

    filename=$(basename $ds)
    prefix=${filename%.*}
    echo $prefix

    python $code/compute_embeddings.py --model-name intfloat/multilingual-e5-large-instruct --dataset $ds --output-path benchmarks/embeddings-e5/$prefix --batch 20 --field-to-encode benchmark_text

done
