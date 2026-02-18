#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=embed
#SBATCH --output=logs/embed-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --cores 4
#SBATCH --mem=40GB
#SBATCH --account=project_2000539

module load pytorch

mkdir -p logs

export HF_HOME="/scratch/project_2000539/jenna/hf-cache"
export HF_DATASETS_CACHE="/scratch/project_2000539/jenna/hf-cache"


prefix=$1

echo $prefix

code="/scratch/project_2000539/jenna/git_checkout/embedding-prompt-analysis"

echo "$code/compute_embeddings.py"

python $code/compute_embeddings.py --model-name intfloat/multilingual-e5-large-instruct --dataset samples/$prefix-fin-10K-sample.jsonl.gz --output-path samples/embeddings-e5/$prefix-fin-10K-sample --batch 20

python $code/compute_embeddings.py --model-name intfloat/multilingual-e5-large-instruct --dataset samples/$prefix-fin-1M-sample.jsonl.gz --output-path samples/embeddings-e5/$prefix-fin-1M-sample --batch 20

##python $code/compute_embeddings.py --model-name intfloat/multilingual-e5-large-instruct --dataset samples/$prefix-fin-5M-sample.jsonl.gz --output-path samples/embeddings-e5/$prefix-fin-5M-sample --batch 20
