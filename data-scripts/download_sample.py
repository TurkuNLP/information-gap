from datasets import load_dataset
import json
import gzip
import argparse
import random
import sys

def download_and_sample(args):

    sampled = []
    counter = 0
    sample_size = 0

    dataset = load_dataset(args.dataset_name, args.language_code, split=args.split, streaming=True, cache_dir=args.cache_dir)

    # go through the dataset and do initial sampling
    initial_sample_percentage = args.sample_size * 2.0 / 150000000 # keep twice as much as needed for final
    for example in dataset:
        counter += 1
        if random.random() < initial_sample_percentage: # keep enough but not too much
            sample_size += 1
            sampled.append(example)

        if counter % 1000000 == 0:
            print(f"Read {counter:,} lines, sampled {sample_size:,} documents", file=sys.stderr, flush=True)

    print(f"{len(sampled):,} documents in the initial sample", file=sys.stderr, flush=True)

    final_sample = random.sample(sampled, args.sample_size)
    print(f"{len(final_sample):,} documents in the final sample", file=sys.stderr, flush=True)

    with gzip.open(args.output_file, "wt", encoding="utf-8") as f:
        for example in final_sample:
            print(json.dumps(example, ensure_ascii=False), file=f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, choices=["MultiSynt/MT-Nemotron-CC", "Helsinki-NLP/nemotron-cc-translated"])
    parser.add_argument("--language-code", type=str, required=True, help="Language code for the dataset, e.g. 'fin_Latn' for MultiSynt/MT-Nemotron-CC or 'fin' for Helsinki-NLP/nemotron-cc-translated.")
    parser.add_argument("--split", type=str, required=True, choices=["tower9b_all", "tower72b_all", "train"], help="Split to download; 'tower9b_all' or 'tower72b_all' for MultiSynt/MT-Nemotron-CC, 'train' for Helsinki-NLP/nemotron-cc-translated.")
    parser.add_argument("--output-file", type=str, required=True, help="Output file name for the sampled dataset.")
    parser.add_argument("--cache-dir", type=str, default="/scratch/project_2000539/jenna/hf-cache", help="HF cache directory for the dataset.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of examples to sample from the dataset.")
    args = parser.parse_args()

    download_and_sample(args)

    # Example usage: python download_sample.py --dataset-name MultiSynt/MT-Nemotron-CC --language-code por_Latn --split tower9b_all --output-file ../../../../information-gap-data/samples/translated-tower9b-por-1M-sample.jsonl.gz --sample-size 1000000

