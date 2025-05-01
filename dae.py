import sys
import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import random


class DAE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2, n_input, n_z):
    # def __init__(self, n_enc_1, n_dec_1, n_input, n_z):
        super(DAE, self).__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        #
        self.z_layer = Linear(n_enc_2, n_z)
        #
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)

        self.x_bar_layer = Linear(n_dec_2, n_input)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))

        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, z

    def add_noise(self, x):
        return x + torch.randn(x.shape)


class TriAE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2, n_input, n_z):
    # def __init__(self, n_enc_1, n_dec_1, n_input, n_z):
        super(TriAE, self).__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        #
        self.z_layer = Linear(n_enc_2, n_z)
        #
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)

        self.x_bar_layer = Linear(n_dec_2, n_input)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))

        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return z, x_bar


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded