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
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

ONEHOT = torch.cat((
    torch.ones(1, 4) / 4, # 
    torch.eye(4), # A, C, G, T
    torch.zeros(1, 4), # padding
), dim=0).float()

class ConvTower(nn.Module):
    def __init__(self, in_channel, out_channel: int, kernel_size) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channel),
            nn.MaxPool1d(kernel_size=kernel_size//2),
            nn.GELU(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class scBasset(nn.Module):
    def __init__(self, n_cells: int, batch_ids: Optional[Iterable[int]]=None, hidden_size=32, seq_len: int=1344, bias: bool=True):
        super().__init__()
        self.config = {
            "n_cells": n_cells,
            "hidden_size": hidden_size,
            "seq_len": seq_len
        }
        if batch_ids is None:
            self.batch_ids = None
        else:
            self.batch_embedding = nn.Embedding(max(batch_ids) + 1, hidden_size)
            self.batch_ids = nn.Parameter(torch.as_tensor(batch_ids), requires_grad=False)
            assert self.batch_ids.ndim == 1
        # self.onehot = nn.Parameter(ONEHOT, requires_grad=False)
        self.register_buffer("onehot", ONEHOT)
        self.seq_len = seq_len

        current_len = seq_len
        self.pre_conv = nn.Sequential( # input: (batch_size, 4, seq_len)
            nn.Conv1d(4, out_channels=288, kernel_size=17, padding=8),
            nn.BatchNorm1d(288),
            nn.MaxPool1d(kernel_size=3), # output: (batch_size, 288, 448)
            nn.GELU(),
        )
        current_len = current_len // 3

        # 2
        kernel_nums = [288, 288, 323, 363, 407, 456, 512]
        self.conv_towers = []
        for i in range(1, 7):
            self.conv_towers.append(ConvTower(kernel_nums[i - 1], kernel_nums[i], kernel_size=5))
            current_len = current_len // 2 # 448 -> 224 -> 112 -> 56 -> 28 -> 14 -> 7; (batch_size, 512, 7)
        self.conv_towers = nn.Sequential(*self.conv_towers)

        # 3
        self.post_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.GELU(), # (batch_size, 256, 7)
        )
        current_len = current_len // 1

        # 4
        self.flatten = nn.Flatten() # (batch_size, 1792)

        current_len = current_len * 256

        # 5
        self.dense = nn.Sequential(
            nn.Linear(current_len, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # 6 
        self.cell_embedding = nn.Linear(hidden_size, n_cells, bias=bias)
        # self.cell_embedding = nn.Parameter(torch.randn(hidden_size, n_cells)) # (hidden_size, n_cells)
        # self.cell_bias = nn.Parameter(torch.randn(n_cells))
    
    def get_embedding(self):
        return self.cell_embedding.state_dict()["weight"]
    
    def forward(self, sequence: Tensor) -> Tensor:
        r"""
        sequence: (batch_size, seq_len), one-hot encoded sequence, 0: N, 1: A, 2: C, 3: G, 4: T, -1: padding
        """
        # assert sequence.shape[1] == self.seq_len
        step = 0
        sequence = self.onehot[sequence.long()].transpose(1, 2)
        sequence = self.pre_conv(sequence)
        sequence = self.conv_towers(sequence)
        sequence = self.post_conv(sequence)
        sequence = self.flatten(sequence)
        sequence = self.dense(sequence) # (B, hidden_size)
        logits = self.cell_embedding(sequence)
        
        # print(self.cell_embedding.weight.shape,self.cell_embedding.bias.shape)
        lr_reg_cell = torch.norm(self.cell_embedding.weight, p=2) + torch.norm(self.cell_embedding.bias, p=2)
        
        if self.batch_ids is not None:
            batch_embed = self.batch_embedding(self.batch_ids).T # (n_cell, hidden_size) -> (hidden_size, n_cell)
            # logits += torch.matmul(sequence, batch_embed)
            lr_reg_batch = torch.norm(self.batch_embedding.weight, p=2) ####print(加basi)
        else:
            lr_reg_batch = 0
        
        return logits, lr_reg_cell, lr_reg_batch

        # return self.cell_embedding(self.dense(sequence))


# class scBasset(nn.Module):
#     def __init__(self, n_cells: int, batch_ids: Optional[Iterable[int]]=None, hidden_size=32, seq_len: int=1344, bias: bool=True):
#         super().__init__()
#         self.config = {
#             "n_cells": n_cells,
#             "hidden_size": hidden_size,
#             "seq_len": seq_len
#         }
#         if batch_ids is None:
#             self.batch_ids = None
#         else:
#             self.batch_embedding = nn.Embedding(max(batch_ids) + 1, hidden_size)
#             self.batch_ids = nn.Parameter(torch.as_tensor(batch_ids), requires_grad=False)
#             assert self.batch_ids.ndim == 1
#         # self.onehot = nn.Parameter(ONEHOT, requires_grad=False)
#         self.register_buffer("onehot", ONEHOT)
#         self.seq_len = seq_len

#         current_len = seq_len
#         self.pre_conv = nn.Sequential( # input: (batch_size, 4, seq_len)
#             nn.Conv1d(4, out_channels=288, kernel_size=17, padding=8),
#             nn.BatchNorm1d(288),
#             nn.MaxPool1d(kernel_size=3), # output: (batch_size, 288, 448)
#             nn.GELU(),
#         )
#         current_len = current_len // 3

#         # 2
#         kernel_nums = [288, 288, 323, 363, 407, 456, 512]
#         self.conv_towers = []
#         for i in range(1, 7):
#             self.conv_towers.append(ConvTower(kernel_nums[i - 1], kernel_nums[i], kernel_size=5))
#             current_len = current_len // 2 # 448 -> 224 -> 112 -> 56 -> 28 -> 14 -> 7; (batch_size, 512, 7)
#         self.conv_towers = nn.Sequential(*self.conv_towers)

#         # 3
#         self.post_conv = nn.Sequential(
#             nn.Conv1d(512, 256, kernel_size=1),
#             nn.BatchNorm1d(256),
#             nn.GELU(), # (batch_size, 256, 7)
#         )
#         current_len = current_len // 1

#         # 4
#         self.flatten = nn.Flatten() # (batch_size, 1792)

#         current_len = current_len * 256

#         # 5
#         self.dense = nn.Sequential(
#             nn.Linear(current_len, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.GELU(),
#             nn.Dropout(0.2),
#         )

#         # 6 
#         self.cell_embedding = nn.Linear(hidden_size, n_cells, bias=bias)
#         # self.cell_embedding = nn.Parameter(torch.randn(hidden_size, n_cells)) # (hidden_size, n_cells)
#         # self.cell_bias = nn.Parameter(torch.randn(n_cells))
    
#     def get_embedding(self):
#         return self.cell_embedding.state_dict()["weight"]
    
#     def forward(self, sequence: Tensor) -> Tensor:
#         r"""
#         sequence: (batch_size, seq_len), one-hot encoded sequence, 0: N, 1: A, 2: C, 3: G, 4: T, -1: padding
#         """
#         # assert sequence.shape[1] == self.seq_len
#         step = 0
#         sequence = self.onehot[sequence.long()].transpose(1, 2);print(step,sequence.shape);step+=1
#         sequence = self.pre_conv(sequence);print(step,sequence.shape);step+=1
#         sequence = self.conv_towers(sequence);print(step,sequence.shape);step+=1
#         sequence = self.post_conv(sequence);print(step,sequence.shape);step+=1
#         sequence = self.flatten(sequence);print(step,sequence.shape);step+=1
#         sequence = self.dense(sequence);print(step,sequence.shape);step+=1 # (B, hidden_size)
#         logits = self.cell_embedding(sequence);print(step,logits.shape)
        
#         print(self.cell_embedding.weight.shape,self.cell_embedding.bias.shape)
#         lr_reg_cell = torch.norm(self.cell_embedding.weight, p=2) + torch.norm(self.cell_embedding.bias, p=2)
        
#         if self.batch_ids is not None:
#             batch_embed = self.batch_embedding(self.batch_ids).T # (n_cell, hidden_size) -> (hidden_size, n_cell)
#             # logits += torch.matmul(sequence, batch_embed)
#             lr_reg_batch = torch.norm(self.batch_embedding.weight, p=2)
#         else:
#             lr_reg_batch = 0
        
#         return logits, lr_reg_cell, lr_reg_batch

#         # return self.cell_embedding(self.dense(sequence))
    