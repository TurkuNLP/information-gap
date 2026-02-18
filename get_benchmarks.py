import argparse
import json
import sys
import os
from collections import Counter
from datasets import load_dataset, load_dataset_builder

# //////////////////////////////////////////////////////////////

def process_arc(example):
    example["benchmark_text"] = example["question"]
    return example

def process_belebele(example):
    example["benchmark_text"] = example["flores_passage"] + " " + example["question"]
    # convert datetime to string
    example["ds"] = example["ds"].isoformat()
    return example

def process_goldenswag(example):
    example["benchmark_text"] = example["ctx"]
    return example

def process_truthfulqa(example):
    example["benchmark_text"] = example["question"]
    return example

def process_scandisent(example):
    example["benchmark_text"] = example["text"]
    return example

def process_squad(example):
    example["benchmark_text"] = example["context"] + " " + example["question"]
    return example

def process_sib(example):
    example["benchmark_text"] = example["text"]
    return example

# //////////////////////////////////////////////////////////////

mapping = {
    "TurkuNLP/finbenchv2-arc-c-fi-ht": process_arc, # mc
    "TurkuNLP/finbenchv2-belebele-fi-og": process_belebele, # reading comprehension mc
    "TurkuNLP/finbenchv2-goldenswag-fi-ht": process_goldenswag, # ??? TODO
    "TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt": process_truthfulqa, # ??? TODO mc with multiple correct and incorrect answers
    "TurkuNLP/finbenchv2-scandisent-fi-mini": process_scandisent, # sentiment analysis (binary)
    "TurkuNLP/finbenchv2-squad-strip-fi-mt": process_squad, # question answering (span/generative)
    "TurkuNLP/finbenchv2-sib-200-fi-og": process_sib # sentence classification (multiclass)
}

# subsets to select
subsets = {"TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt": "mc_FI"}


# Prefer this split order when "test" is not available (only one split is downloaded)
DEFAULT_SPLIT_ORDER = ("test", "validation", "eval", "train")


def _choose_split(benchmark_name, subset):
    """Get available splits without loading data; return preferred single split name."""
    if subset:
        builder = load_dataset_builder(benchmark_name, subset)
    else:
        builder = load_dataset_builder(benchmark_name)
    available = list(builder.info.splits.keys())
    for name in DEFAULT_SPLIT_ORDER:
        if name in available:
            return name
    if not available:
        raise ValueError(f"Dataset {benchmark_name} has no splits")
    return available[0]


def get_benchmark_data(benchmark_name):
    # define split and subset
    subset = subsets[benchmark_name] if benchmark_name in subsets else None
    split = _choose_split(benchmark_name, subset)
    dataset = load_dataset(benchmark_name, subset, split=split) if subset else load_dataset(benchmark_name, split=split)
    print(dataset.info.dataset_name, dataset.info.splits[split], file=sys.stderr)
    examples = [e for e in dataset]
    examples = [mapping[benchmark_name](example) for example in examples]
    return examples

def save_benchmark(examples, filename):
    with open(filename, 'w') as f:
        for example in examples:
            #print(type(example), example)
            print(json.dumps(example, ensure_ascii=False), file=f)


def main(args):
    for benchmark in args.benchmarks:
        examples = get_benchmark_data(benchmark)
        output_file = f"benchmarks/{benchmark.replace('/', '-')}.jsonl"
        save_benchmark(examples, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', type=str, nargs='+', default=["TurkuNLP/finbenchv2-arc-c-fi-ht", "TurkuNLP/finbenchv2-belebele-fi-og", "TurkuNLP/finbenchv2-goldenswag-fi-ht", "TurkuNLP/finbenchv2-opengpt-x_truthfulqax-fi-mt", "TurkuNLP/finbenchv2-scandisent-fi-mini", "TurkuNLP/finbenchv2-squad-strip-fi-mt", "TurkuNLP/finbenchv2-sib-200-fi-og"])
    args = parser.parse_args()
    main(args)