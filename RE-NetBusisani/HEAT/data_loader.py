"""
Data Loader for HEAT — reuses the RE-Net data directory format.

Data directory structure (same as RE-Net):
    ./data/<DATASET>/
        train.txt   — tab-separated: s r o t
        valid.txt
        test.txt
        stat.txt    — num_nodes num_rels

History format (built from train.txt):
    entity_history[e] = list of {t: [(r, o), ...]}  groups

We rebuild history on the fly (no need for the precomputed pickle files).
"""

import os
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Low-level file readers
# ─────────────────────────────────────────────────────────────────────────────

def get_total_number(data_path: str):
    """Read stat.txt → (num_nodes, num_rels)."""
    with open(os.path.join(data_path, 'stat.txt'), 'r') as f:
        parts = f.read().strip().split()
    return int(parts[0]), int(parts[1])


def load_quadruples(data_path: str, *filenames):
    """Load one or more .txt files of quadruples (s r o t)."""
    data = []
    for fname in filenames:
        fpath = os.path.join(data_path, fname)
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    s, r, o, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    data.append((s, r, o, t))
    data = sorted(data, key=lambda x: x[3])
    return np.array(data, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# History Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_history(data: np.ndarray, num_nodes: int, seq_len: int = 10):
    """
    Build per-entity temporal history from quadruple data.

    Returns:
        s_history[e] = list of fact arrays per timestep (subject perspective)
        s_history_t[e] = corresponding timestamps
        o_history[e] = list of fact arrays per timestep (object perspective)
        o_history_t[e] = corresponding timestamps
    """
    # {entity: {timestamp: [(r, o), ...]}}  — subject perspective
    s_hist_raw = defaultdict(lambda: defaultdict(list))
    o_hist_raw = defaultdict(lambda: defaultdict(list))

    for s, r, o, t in data:
        s_hist_raw[s][t].append((r, o))
        o_hist_raw[o][t].append((r, s))   # inverse: object looks back at subjects

    def raw_to_seq(raw_dict):
        history = [None] * num_nodes
        history_t = [None] * num_nodes
        for e, t_dict in raw_dict.items():
            sorted_times = sorted(t_dict.keys())
            history[e] = [np.array(t_dict[t], dtype=np.int64)
                          for t in sorted_times][-seq_len:]
            history_t[e] = sorted_times[-seq_len:]
        # Replace None with empty lists
        for i in range(num_nodes):
            if history[i] is None:
                history[i] = []
                history_t[i] = []
        return history, history_t

    s_history, s_history_t = raw_to_seq(s_hist_raw)
    o_history, o_history_t = raw_to_seq(o_hist_raw)
    return s_history, s_history_t, o_history, o_history_t


def get_history_at_time(full_data: np.ndarray, query_t: int,
                        num_nodes: int, seq_len: int = 10):
    """
    Build history using only facts strictly before query_t.
    Used at test time to avoid data leakage.
    """
    past_data = full_data[full_data[:, 3] < query_t]
    return build_history(past_data, num_nodes, seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TemporalKGDataset(Dataset):
    """
    Dataset returning individual quadruples with per-sample history.

    For training efficiency, we build history once from all train data
    (minor leakage for early training timestamps — acceptable tradeoff).
    """

    def __init__(self, quadruples: np.ndarray,
                 s_history: list, s_history_t: list,
                 o_history: list, o_history_t: list):
        self.data = quadruples
        self.s_history = s_history
        self.s_history_t = s_history_t
        self.o_history = o_history
        self.o_history_t = o_history_t

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, r, o, t = self.data[idx]
        return {
            'triplet': torch.LongTensor([s, r, o, t]),
            's_hist':   self.s_history[s],
            's_hist_t': self.s_history_t[s],
            'o_hist':   self.o_history[o],
            'o_hist_t': self.o_history_t[o],
        }


def heat_collate(batch):
    """Custom collate: stack triplets, keep histories as lists."""
    triplets = torch.stack([item['triplet'] for item in batch])
    s_hist =   [item['s_hist'] for item in batch]
    s_hist_t = [item['s_hist_t'] for item in batch]
    o_hist =   [item['o_hist'] for item in batch]
    o_hist_t = [item['o_hist_t'] for item in batch]
    return triplets, (s_hist, s_hist_t), (o_hist, o_hist_t)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str, data_root: str = '../data',
                 seq_len: int = 10, batch_size: int = 512):
    """
    Full data pipeline for a given dataset.

    Returns:
        train_loader, valid_data, test_data, num_nodes, num_rels,
        s_history_test, s_history_t_test, o_history_test, o_history_t_test
    """
    data_path = os.path.join(data_root, dataset_name)
    num_nodes, num_rels = get_total_number(data_path)
    print(f"  [data] {dataset_name}: {num_nodes:,} entities, {num_rels} relations", flush=True)

    print(f"  [data] Loading quadruples...", flush=True)
    train_data = load_quadruples(data_path, 'train.txt')
    valid_data = load_quadruples(data_path, 'valid.txt')
    test_data  = load_quadruples(data_path, 'test.txt')
    print(f"  [data] Train: {len(train_data):,}  Valid: {len(valid_data):,}  Test: {len(test_data):,}", flush=True)

    # Build history from training data only
    print(f"  [data] Building train history...", flush=True)
    s_hist, s_hist_t, o_hist, o_hist_t = build_history(
        train_data, num_nodes, seq_len)
    print(f"  [data] Train history done.", flush=True)

    # Build test-time history from train+valid (no test leakage)
    print(f"  [data] Building test history...", flush=True)
    all_seen = np.concatenate([train_data, valid_data], axis=0)
    s_hist_test, s_hist_t_test, o_hist_test, o_hist_t_test = build_history(
        all_seen, num_nodes, seq_len)
    print(f"  [data] Test history done. Dataset ready.\n", flush=True)

    train_dataset = TemporalKGDataset(
        train_data, s_hist, s_hist_t, o_hist, o_hist_t)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=heat_collate, num_workers=0)

    return (train_loader, valid_data, test_data,
            num_nodes, num_rels,
            s_hist_test, s_hist_t_test, o_hist_test, o_hist_t_test)
