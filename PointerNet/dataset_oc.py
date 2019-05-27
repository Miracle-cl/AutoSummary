import torch
import numpy as np
from torch.utils.data import Dataset

def collate_fn(insts):
    # if seq_pad in class then all seqs with same length
    # <PAD> - 929
    maxlen = max([len(x) for x in insts])
    seq = np.array([x + [929] * (maxlen - len(x)) for x in insts])
    seq_lens = np.array([len(x) for x in insts])
    return torch.LongTensor(seq), torch.LongTensor(seq_lens)

def paired_collate_fn(insts):
    #src_insts, tgt_insts = list(zip(*insts))
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, tgt_insts, tgt_ids = zip(*seq_pairs)
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    tgt_ids = collate_fn(tgt_ids)
    return (*src_insts, *tgt_insts, *tgt_ids)

class OutcomeDatasets(Dataset):
    def __init__(self, src, tgt, tgt_ids):
        self.src = src
        self.tgt = tgt
        self.tgt_ids = tgt_ids

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.tgt_ids[idx]
