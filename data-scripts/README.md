## Download and sample MultiSynt data

Download from HuggungFace and sample

`python download_sample.py --dataset-name MultiSynt/MT-Nemotron-CC --language-code por_Latn --split tower9b_all --output-file ../../../../information-gap-data/samples/translated-tower9b-por-1M-sample.jsonl.gz --sample-size 1000000`

Sample

`zcat translated-tower9b-por-1M-sample.jsonl.gz | python sample.py --final-sample-size 10000 | gzip -c > translated-tower9b-por-10K-sample.jsonl.gz`

Use `--initial-sample-percentage` to reduce memory usage, e.g. 0.1 keeps random 10% of documents to do the final sample (default: keep all). Set the value based on original file size and final sample size to keep enough documents to do final sampling.

## Download and process benchmark datasets

TODO