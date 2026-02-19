# code to print documents based in cluster index

import json
import sys
import argparse
import os
from collections import Counter
import gzip

def read_original_files(args):
    original_files = {}
    for original_file in args.original_files:
        print(f"Reading original file: {original_file}", file=sys.stderr)
        with gzip.open(original_file, 'rt') if original_file.endswith(".gz") else open(original_file, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                # get base name
                fname = os.path.basename(original_file)
                if fname not in original_files:
                    original_files[fname] = {}
                original_files[fname][i] = data

    return original_files

def yield_documents(args):
    for document_file in args.cluster_metadata:
        print(f"Reading cluster metadata file: {document_file}", file=sys.stderr)
        with open(document_file, 'r') as f:
            for line in f:
                yield json.loads(line)


def print_document_text(document, original_files):
    metadata = document["metadata"]
    origin = os.path.basename(metadata["origin"].replace(".0.examples.jsonl", ".jsonl").replace(".0.examples.jsonl.gz", ".jsonl.gz"))
    
    fname = os.path.basename(origin)
    index = metadata["global_example_index"]
    data = original_files[fname][index]
    print(f"\n------------------------------\nDocument {index} from {fname}")
    if "text" in data:
        print(data["text"],"\n\n", file=sys.stdout)
    elif "question" in data:
        print(data["question"],"\n\n", file=sys.stdout)
    else:
        print(data,"\n\n", file=sys.stdout)


def main(args):


    original_files = read_original_files(args)

    total_documents = 0
    labels = Counter()
    for document in yield_documents(args):
        if document["cluster_index"] == args.cluster_index:
            total_documents += 1
            labels.update([document["metadata"]["label"]])
            print_document_text(document, original_files)

    print(f"\nTotal documents in cluster {args.cluster_index}: {total_documents}", file=sys.stdout)
    print(f"Labels: {labels.most_common(10)}", file=sys.stdout)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster-metadata', type=str, nargs='+', required=True)
    parser.add_argument('--original-files', type=str, nargs='+', required=True)
    parser.add_argument('--cluster-index', type=int, required=True)
    args = parser.parse_args()
    main(args)