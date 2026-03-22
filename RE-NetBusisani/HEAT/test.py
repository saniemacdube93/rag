"""
HEAT Test/Evaluation Script

Computes full filtered MRR, Hits@1, Hits@3, Hits@10 on the test set.
Also collects raw scores and uncertainty estimates needed for visualization.

Usage:
    python test.py -d ICEWS18
"""

import argparse
import json
import os
import numpy as np
import torch

from model.heat_model import HEATModel
from data_loader import load_dataset, get_total_number, load_quadruples

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_full(model, test_data, s_hist, s_hist_t, o_hist, o_hist_t,
                  all_data, num_nodes, compute_uncertainty=False):
    """
    Full filtered evaluation returning ranks, scores, and uncertainty.
    Returns dict with all metrics + raw data for visualisation.
    """
    model.eval()

    # Build filter set
    gt = {}
    for s, r, o, t in all_data:
        gt.setdefault((int(s), int(r), int(t)), set()).add(int(o))
        gt.setdefault((int(o), int(r), int(t)), set()).add(int(s))  # inverse key slightly different

    ranks_all = []
    top_preds = []          # (pred_entity, true_entity) pairs
    uncertainty_vals = []   # per-query uncertainty

    print(f"Evaluating on {len(test_data)} test quadruples...")

    with torch.no_grad():
        for idx, (s, r, o, t) in enumerate(test_data):
            s, r, o, t = int(s), int(r), int(o), int(t)

            for direction in ['subject', 'object']:
                if direction == 'subject':
                    src, tgt = s, o
                    hist = s_hist[src]
                    hist_t = s_hist_t[src]
                    subject = True
                    fkey = (src, r, t)
                else:
                    src, tgt = o, s
                    hist = o_hist[src]
                    hist_t = o_hist_t[src]
                    subject = False
                    fkey = (src, r, t)

                if compute_uncertainty:
                    probs, unc = model.predict_with_uncertainty(
                        src, r, float(t), hist, hist_t, subject)
                    scores = probs.cpu().numpy()
                    uncertainty_vals.append(float(unc[tgt].item()))
                else:
                    logits = model.predict_scores(
                        src, r, float(t), hist, hist_t, subject)
                    scores = logits.cpu().numpy()

                target_score = scores[tgt]

                # Apply filter
                scores_filtered = scores.copy()
                for ans in gt.get(fkey, set()):
                    if ans != tgt:
                        scores_filtered[ans] = -1e9

                rank = int(np.sum(scores_filtered > target_score) + 1)
                ranks_all.append(rank)

                # Store top prediction
                top_pred = int(np.argmax(scores_filtered))
                top_preds.append((top_pred, tgt))

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(test_data)}")

    ranks = np.array(ranks_all)
    mrr   = float(np.mean(1.0 / ranks))
    h1    = float(np.mean(ranks <= 1))
    h3    = float(np.mean(ranks <= 3))
    h10   = float(np.mean(ranks <= 10))
    mr    = float(np.mean(ranks))

    results = {
        'mrr':   mrr,
        'mr':    mr,
        'hits1': h1,
        'hits3': h3,
        'hits10': h10,
        'ranks': ranks.tolist(),
        'top_preds': top_preds,
        'uncertainty': uncertainty_vals,
    }
    return results


def main(args):
    data_path = os.path.join(args.data_root, args.dataset)
    num_nodes, num_rels = get_total_number(data_path)

    train_data = load_quadruples(data_path, 'train.txt')
    valid_data = load_quadruples(data_path, 'valid.txt')
    test_data  = load_quadruples(data_path, 'test.txt')
    all_data   = np.concatenate([train_data, valid_data, test_data], axis=0)

    from data_loader import build_history
    all_seen = np.concatenate([train_data, valid_data], axis=0)
    s_hist, s_hist_t, o_hist, o_hist_t = build_history(
        all_seen, num_nodes, seq_len=args.seq_len)

    # Load model
    checkpoint = torch.load(
        f'results/{args.dataset}/heat_best.pth',
        map_location=_device)
    saved_args = checkpoint.get('args', {})

    model = HEATModel(
        num_nodes=num_nodes,
        num_rels=num_rels,
        h_dim=saved_args.get('n_hidden', args.n_hidden),
        time_dim=saved_args.get('time_dim', 32),
        n_local_heads=saved_args.get('n_local_heads', 4),
        n_global_heads=saved_args.get('n_global_heads', 4),
        seq_len=saved_args.get('seq_len', args.seq_len),
        dropout=0.0,   # No dropout at test time for point estimates
        mc_samples=args.mc_samples,
    ).to(_device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"\n{'='*60}")
    print(f"  HEAT Evaluation — {args.dataset}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"{'='*60}\n")

    results = evaluate_full(
        model, test_data, s_hist, s_hist_t, o_hist, o_hist_t,
        all_data, num_nodes,
        compute_uncertainty=args.uncertainty,
    )

    # Print results table
    print(f"\n{'─'*40}")
    print(f"  Test Results — HEAT / {args.dataset}")
    print(f"{'─'*40}")
    print(f"  MRR   : {results['mrr']:.4f}")
    print(f"  MR    : {results['mr']:.1f}")
    print(f"  Hits@1: {results['hits1']:.4f}")
    print(f"  Hits@3: {results['hits3']:.4f}")
    print(f"  Hits@10:{results['hits10']:.4f}")
    print(f"{'─'*40}\n")

    # Save
    os.makedirs(f'results/{args.dataset}', exist_ok=True)
    out_path = f'results/{args.dataset}/test_results.json'
    save_results = {k: v for k, v in results.items()
                    if k not in ('ranks', 'top_preds', 'uncertainty')}
    save_results['ranks'] = results['ranks'][:100]  # truncate for file size
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save full results for visualisation
    import pickle
    with open(f'results/{args.dataset}/eval_data.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to results/{args.dataset}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='../data')
    parser.add_argument('--n-hidden', type=int, default=200)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--mc-samples', type=int, default=20)
    parser.add_argument('--uncertainty', action='store_true',
                        help='Enable MC dropout uncertainty estimation')
    args = parser.parse_args()
    main(args)
