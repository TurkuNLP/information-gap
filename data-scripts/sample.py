import json
import random
import sys
import argparse

def main(args):

    sampled = []

    counter = 0
    sample_size = 0
    for line in sys.stdin:
        counter += 1
        
        if random.random() < args.initial_sample_percentage: # keep enough but not too much
            sample_size += 1
            sampled.append(line)

        if counter % 1000000 == 0:
            print(f"Read {counter:,} lines, sampled {sample_size:,} documents", file=sys.stderr, flush=True)

    print(f"{len(sampled):,} documents in the initial sample", file=sys.stderr, flush=True)

    final_sample = random.sample(sampled, args.final_sample_size)

    print(f"{len(final_sample):,} documents in the final sample", file=sys.stderr, flush=True)

    for example in final_sample:
        print(example, end="")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-sample-percentage", type=float, default=1.0, help="Percentage to keep in the initial sample for final sampling, default is 1.0 (keep all). Use smaller values to save memory.")
    parser.add_argument("--final-sample-size", type=int, default=10000, help="Number of examples to sample from the dataset.")
    args = parser.parse_args()

    main(args)