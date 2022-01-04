import numpy as np
import torch
from torch.utils.data import Dataset

data = open('data.txt').read()
vocab = sorted((set(data)))
vocab_size = len(vocab)

idx_to_char = {i:c for i,c in enumerate(vocab)}
char_to_idx = {c:i for i,c in enumerate(vocab)}

def text_to_idx(text):
    out = [char_to_idx[i] for i in text]
    return out

def hot_encode(text):
    out = text_to_idx(text)
    out = torch.eye(vocab_size)[out]

    return out

def decode_idxs(idxs):
    return ''.join([idx_to_char[i] for i in idxs])


class Data(Dataset):
    def __init__(self,train=True,seq_len=100):
        self.seq_len = seq_len

        if train:
            self.data = data[:int(.9*len(data))]

        else:
            self.data = data[int(.9*len(data)):]
        
    def __len__(self):
        return len(self.data)-self.seq_len-1

    def __getitem__(self, idx):
        x = self.data[idx:self.seq_len+idx]
        y = self.data[idx+1:self.seq_len+idx+1]

        x = hot_encode(x)
        y = torch.Tensor(text_to_idx(y))

        return x,y
