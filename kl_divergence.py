import argparse
from analyze_clusters import read_documents, get_cluster_labels
import numpy as np
# from scipy.special import rel_entr, kl_div 



def prob_distribution(cluster_labels, label):
    e = 1e-8  # small epsilon to avoid zero in counts and in KL
    x = []
    for i in range(len(cluster_labels.keys())):
        labels = cluster_labels[i]
        count = labels.count(label)
        x.append(count + e)  # add epsilon to every count so no zeros
    # normalize to a probability distribution
    x = np.array(x, dtype=float)
    p = x / x.sum()
    return np.clip(p, e, 1.0)  # clip to safety range [e, 1.0], should not do anything


def KL(P, Q):
    # verified to match both scipy's np.sum(rel_entr(P, Q)) and np.sum(kl_div(P, Q))
    divergence = np.sum(P*np.log(P/Q))
    return divergence



def main(args):

    documents = read_documents(args.clustered_documents)
    cluster_labels = get_cluster_labels(documents)
    print(f"Total labels {args.label1}: {len([l for i,labels in cluster_labels.items() for l in labels if l == args.label1])}")
    print(f"Total labels {args.label2}: {len([l for i,labels in cluster_labels.items() for l in labels if l == args.label2])}")

    prob_distribution1 = prob_distribution(cluster_labels, args.label1)
    prob_distribution2 = prob_distribution(cluster_labels, args.label2)

    kl = KL(prob_distribution1, prob_distribution2)
    kl_rev = KL(prob_distribution2, prob_distribution1)
    print(f"KL divergence")
    print(f"P: {args.label1}, Q: {args.label2} → KL: {kl}")
    print(f"P: {args.label2}, Q: {args.label1} → KL: {kl_rev}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--clustered-documents', type=str, nargs='+', default=['toy_data/toy_documents.jsonl'])
    parser.add_argument('--label1', type=str, default="native", help="Label 1")
    parser.add_argument('--label2', type=str, default="tropus", help="Label 2")
    args = parser.parse_args()
    main(args)