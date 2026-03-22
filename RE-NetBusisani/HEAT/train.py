"""
HEAT Training Script
Single-stage joint training — no pretrain step needed.

Usage:
    python train.py -d ICEWS18 --n-hidden 200 --max-epochs 30 --batch-size 512

Results saved to: results/<DATASET>/
"""

import argparse
import os
import time
import json
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.heat_model import HEATModel
from losses.losses import HEATLoss
from data_loader import load_dataset

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: HEATModel, data: np.ndarray,
             s_hist: list, s_hist_t: list,
             o_hist: list, o_hist_t: list,
             all_data: np.ndarray, num_nodes: int,
             max_samples: int = 2000):
    """
    Compute MRR and Hits@1/3/10 (filtered) on a data split.
    Capped at max_samples for speed during training validation.
    """
    model.eval()
    ranks = []
    # Build ground-truth answer set for filtering
    gt = {}
    for s, r, o, t in all_data:
        gt.setdefault((s, r, t), set()).add(o)
        gt.setdefault((o, r + num_nodes, t), set()).add(s)  # inverse

    indices = np.random.permutation(len(data))[:max_samples]

    with torch.no_grad():
        for idx in indices:
            s, r, o, t = data[idx]

            for direction in ['s', 'o']:
                if direction == 's':
                    src, tgt = s, o
                    hist = s_hist[src]
                    hist_t = s_hist_t[src]
                    subject = True
                    filter_key = (src, r, int(t))
                else:
                    src, tgt = o, s
                    hist = o_hist[src]
                    hist_t = o_hist_t[src]
                    subject = False
                    filter_key = (src, r + num_nodes, int(t))

                logits = model.predict_scores(
                    int(src), int(r), float(t), hist, hist_t, subject)
                scores = logits.cpu().numpy()

                target_score = scores[tgt]
                # Filter out other known answers
                for ans in gt.get(filter_key, set()):
                    if ans != tgt:
                        scores[ans] = -1e9

                rank = int(np.sum(scores > target_score) + 1)
                ranks.append(rank)

    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    hits1 = float(np.mean(ranks <= 1))
    hits3 = float(np.mean(ranks <= 3))
    hits10 = float(np.mean(ranks <= 10))
    return mrr, hits1, hits3, hits10


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs(f'results/{args.dataset}', exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  HEAT Training — Dataset: {args.dataset}")
    print(f"  Device: {_device}")
    print(f"{'='*60}\n")

    # Load data
    (train_loader, valid_data, test_data,
     num_nodes, num_rels,
     s_hist_test, s_hist_t_test,
     o_hist_test, o_hist_t_test) = load_dataset(
        args.dataset, data_root=args.data_root,
        seq_len=args.seq_len, batch_size=args.batch_size)

    all_data = np.concatenate([
        train_loader.dataset.data, valid_data, test_data], axis=0)

    print(f"  Entities: {num_nodes:,}  |  Relations: {num_rels}  "
          f"|  Train: {len(train_loader.dataset):,}\n")

    # Build model
    model = HEATModel(
        num_nodes=num_nodes,
        num_rels=num_rels,
        h_dim=args.n_hidden,
        time_dim=args.time_dim,
        n_local_heads=args.n_local_heads,
        n_global_heads=args.n_global_heads,
        seq_len=args.seq_len,
        dropout=args.dropout,
        mc_samples=args.mc_samples,
    ).to(_device)

    loss_fn = HEATLoss(alpha=args.alpha, beta=args.beta,
                       temperature=args.temperature)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)

    best_mrr = 0.0
    history_log = {
        'train_loss': [], 'val_mrr': [],
        'val_hits1': [], 'val_hits3': [], 'val_hits10': [],
        'epoch_time': [],
    }

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = pred_loss = contrast_loss = 0.0
        n_batches = 0

        for triplets, s_hist_batch, o_hist_batch in train_loader:
            triplets = triplets.to(_device)
            optimizer.zero_grad()

            # Forward — subject direction
            loss_s = model(triplets, s_hist_batch, o_hist_batch, subject=True)
            # Forward — object direction (inverse)
            loss_o = model(triplets, s_hist_batch, o_hist_batch, subject=False)
            loss = loss_s + loss_o

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Progress every 10 batches
            if n_batches % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Batch {n_batches}/{len(train_loader)} | "
                      f"Loss {total_loss/n_batches:.4f} | "
                      f"{elapsed:.0f}s elapsed", flush=True)

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        history_log['train_loss'].append(avg_loss)
        history_log['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch:03d}/{args.max_epochs} | "
              f"Loss {avg_loss:.4f} | Time {epoch_time:.1f}s", flush=True)

        # Validation
        if epoch % args.valid_every == 0:
            val_hist = list(zip(*[
                (train_loader.dataset.s_history,
                 train_loader.dataset.s_history_t,
                 train_loader.dataset.o_history,
                 train_loader.dataset.o_history_t)
            ]))[0]
            sh, sht, oh, oht = (train_loader.dataset.s_history,
                                 train_loader.dataset.s_history_t,
                                 train_loader.dataset.o_history,
                                 train_loader.dataset.o_history_t)

            mrr, h1, h3, h10 = evaluate(
                model, valid_data, sh, sht, oh, oht, all_data, num_nodes,
                max_samples=args.val_samples)

            history_log['val_mrr'].append(mrr)
            history_log['val_hits1'].append(h1)
            history_log['val_hits3'].append(h3)
            history_log['val_hits10'].append(h10)

            print(f"  ↳ Valid  MRR: {mrr:.4f} | H@1: {h1:.4f} | "
                  f"H@3: {h3:.4f} | H@10: {h10:.4f}")

            if mrr > best_mrr:
                best_mrr = mrr
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'mrr': mrr,
                    'args': vars(args),
                }, f'results/{args.dataset}/heat_best.pth')
                print(f"  ✓ Best model saved (MRR={mrr:.4f})")

    # Save training history
    with open(f'results/{args.dataset}/training_history.json', 'w') as f:
        json.dump(history_log, f, indent=2)

    print(f"\nTraining done. Best valid MRR: {best_mrr:.4f}")
    print(f"Results saved to results/{args.dataset}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HEAT Training')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='../data')
    parser.add_argument('--n-hidden', type=int, default=200)
    parser.add_argument('--time-dim', type=int, default=32)
    parser.add_argument('--n-local-heads', type=int, default=4)
    parser.add_argument('--n-global-heads', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--grad-norm', type=float, default=1.0)
    parser.add_argument('--valid-every', type=int, default=2)
    parser.add_argument('--val-samples', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='contrastive loss weight')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='temporal consistency loss weight')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='InfoNCE temperature')
    parser.add_argument('--mc-samples', type=int, default=20)
    args = parser.parse_args()
    train(args)
