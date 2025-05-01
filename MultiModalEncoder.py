import argparse
import math
import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
import torch.nn.functional as F
from tqdm import tqdm
# from biock.pytorch import PerformerEncoder, PerformerEncoderLayer, kl_divergence
# from torch.utils.data import DataLoader
import logging
from scbasset import scBasset

def build_mlp(layers, nhead: int=1, activation=nn.ReLU(), bn=True, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        if nhead == 1:
            net.append(nn.Linear(layers[i - 1], layers[i]))
        else:
            net.append(MultiHeadLinear(layers[i - 1], layers[i], nhead=nhead))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class MultiHeadLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, nhead: int, bias: bool=True, device=None, dtype=None) -> None:
        super(MultiHeadLinear, self).__init__()
        assert nhead > 1
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.weight = nn.Parameter(torch.empty(nhead, in_features, out_features)) # (D, H, H')
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, nhead))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        ## input: (B, H) or (B, H, D)
        # return F.linear(input, self.weight, self.bias)
        assert len(input.size()) <= 3
        if len(input.size()) == 3:
            assert input.size(2) == self.weight.size(0), "dimension should be same in MultiHeadLinear, while input: {}, weight: {}".format(input.size(), self.weight.size())
            input = input.transpose(0, 1).transpose(0, 2)
        input = torch.matmul(input, self.weight).transpose(0, 1).transpose(1, 2)
        if self.bias is not None:
            input += self.bias
        return input

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, nhead={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.nhead
        )

class Encoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()
        self.hidden = build_mlp([x_dim]+h_dim, bn=bn, dropout=dropout)
        self.sample = nn.Sequential(
            nn.Linear(([x_dim] + h_dim)[-1], z_dim),
            nn.LeakyReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor

        Return:
            z
        """
        x = self.hidden(x)
        return self.sample(x) ## -> z
    

class Decoder(nn.Module):
    def __init__(self, x_dim: int, h_dim: List[int], z_dim: int, bn=True, dropout=0, output_activation=nn.Sigmoid()):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)

        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        if self.output_activation is not None:
            return self.output_activation(self.reconstruction(x))
        else:
            return self.reconstruction(x)
        

class AE(nn.Module):
    def __init__(self, atac_dim: int, gene_dim: int, h_dim: List[int], z_dim: int):
        super(AE, self).__init__()
        self.atac_en = Encoder(x_dim=atac_dim, h_dim=h_dim, z_dim=z_dim)

        print(h_dim)
        h_dim.reverse()
        print(h_dim)
        self.decoder = Decoder(x_dim=gene_dim, h_dim=h_dim, z_dim=z_dim)

    def forward(self, atac):
        h = self.atac_en(atac)

        out = self.decoder(h)

        return out
        

## Multi modal AE
class MMAE(nn.Module):
    def __init__(self, scbasset: scBasset, atac_dim: int, scemb_dim: int, gene_dim: int, h_dim: List[int], z_dim: int):
        super(MMAE, self).__init__()

        self.emb_gen = scbasset

        self.atac_en = Encoder(x_dim=atac_dim, h_dim=[h_dim[-1]], z_dim=z_dim)
        self.emb_en = Encoder(x_dim=scemb_dim, h_dim=h_dim, z_dim=z_dim)

        h_dim.reverse()
        self.decoder = Decoder(x_dim=gene_dim, h_dim=h_dim, z_dim=z_dim * 2)

    def forward(self, atac, it=None, subset=None, bs=None):
        # emb = self.emb_gen.get_embedding()[subset]
        # # print(emb.shape)
        # emb = emb[(it * bs): ((it + 1) * bs), ]

        emb = self.emb_gen.get_embedding()

        h_s = self.emb_en(emb)
        h_a = self.atac_en(atac)

        h = torch.cat((h_s, h_a), dim=1)

        out = self.decoder(h)

        return out

    def scforward(self, seq):
        return self.emb_gen.forward(seq)


class MMAE2s(nn.Module):
    def __init__(self, atac_dim: int, scemb_dim: int, gene_dim: int, h_dim: List[int], z_dim: int):
        super(MMAE2s, self).__init__()

        self.atac_en = Encoder(x_dim=atac_dim, h_dim=[h_dim[-1]], z_dim=z_dim)
        self.emb_en = Encoder(x_dim=scemb_dim, h_dim=h_dim, z_dim=z_dim)

        h_dim.reverse()
        self.decoder = Decoder(x_dim=gene_dim, h_dim=h_dim, z_dim=z_dim * 2)

    def forward(self, emb, atac):
        h_s = self.emb_en(emb)
        h_a = self.atac_en(atac)

        h = torch.cat((h_s, h_a), dim=1)

        out = self.decoder(h)

        return out