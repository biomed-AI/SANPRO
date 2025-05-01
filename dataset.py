#!/usr/bin/env python3
"""
Author: Ken Chen
Email: chenkenbio@gmail.com
Date: 2023-02-25
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import episcanpy as esp
import h5py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import scanpy as sc
import pandas as pd
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
# from biock import load_fasta, get_reverse_strand, encode_sequence
# from biock import HG38_FASTA_H5
import nums_from_string
import random

import logging
logger = logging.getLogger(__name__)


def load_adata(data, nor=True, hvg=False, log1p=True) -> AnnData:
    if data.split('.')[-1] == 'h5ad':
        adata = sc.read_h5ad(data)
    elif data.split('.')[-1] == 'csv':
        adata = sc.read_csv(data, first_column_names=True).T
    else:
        logger.info('Not implemented!!!')

    if hvg:
        adata = sc.pp.highly_variable_genes(adata, n_top_genes=100)
    if nor:
        sc.pp.normalize_total(adata, target_sum=adata.X.shape[1])
    # sc.pp.normalize_total(adata)
    # print(adata)
    if log1p:
        sc.pp.log1p(adata)
    # print(adata.X)

    return adata

def load_csv(data):
    data = pd.read_csv(data, index_col=0)

    return data


class SingleCellDataset(Dataset):
    def __init__(self, data:AnnData, seq_ref=None, seq_len=1344, batch=None, rmbatch=False, batch_subset=None):
        self.data = data
        if batch is not None:
            self.data = self.data[self.data.obs["donor"] == batch]
        
        if batch_subset is not None:
            index = []
            for item in self.data.obs["donor"]:
                id = 0 # random.randint(0, 4)
                if item in batch_subset and id == 0:
                    index.append(True)
                else:
                    index.append(False)
            self.data = self.data[index]

        self.seq_len = seq_len

        self.obs = self.data.obs.copy()
        del self.data.obs

        self.var = self.data.var.copy()
        del self.data.var

        self.X = csr_matrix(self.data.X.T)
        del self.data.X

        self.ref = seq_ref
        del seq_ref

        self.rmbatch = rmbatch
        if rmbatch:
            self.batche_ids = [int(nums_from_string.get_nums(item)[0]) - 1 for item in self.obs['Batch']]
            # self.batche_ids = np.eye(np.max(self.batche_ids) + 1)[self.batche_ids]
        else:
            self.batche_ids = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        name = self.var.index.values[index]
        # print(name , '   ')

        # if name in self.ref.index.values:
        #     seq = list(self.ref.loc[name].values[0])
        # elif name[:-2] in self.ref.index.values:
        #     seq = list(self.ref.loc[name[:-2]].values[0])
        # print(seq[0:5])
            
        # 随机序列
        ind = random.randint(0, len(self.ref) - 1)
        seq = list(self.ref.iloc[ind].values[0])
        
        atcg2digit = {'A':1, "T":2, "C":3, 'G': 4}
        seq = np.array([atcg2digit[i] for i in seq])

        if len(seq) < self.seq_len:
            seq = np.concatenate((seq, np.full(self.seq_len - len(seq), 0, dtype=seq.dtype)))
        elif len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        

        if self.rmbatch:
            return torch.Tensor(seq), self.X[index].toarray().flatten() # , self.batche_ids[index]
        else:
            return torch.Tensor(seq), self.X[index].toarray().flatten()
        # return seq, self.X[index].toarray().flatten().astype(np.int64)


class AEDataset(Dataset):
    def __init__(self, data:AnnData, batch=None, rmbatch=False):
        self.data = data
        if batch is not None:
            self.data = self.data[self.data.obs["donor"] == batch]

        self.obs = self.data.obs.copy()
        del self.data.obs

        self.var = self.data.var.copy()
        del self.data.var

        self.X = csr_matrix(self.data.X)
        del self.data.X

        self.rm_batch = rmbatch
        if rmbatch:
            self.batche_ids = [int(nums_from_string.get_nums(item)[0]) for item in self.obs['Batch']]
        else:
            self.batche_ids = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # return self.X[index].toarray().flatten(), self.batche_ids[index]
        return self.X[index].toarray().flatten()
    
