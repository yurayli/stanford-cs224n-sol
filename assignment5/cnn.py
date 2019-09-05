#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, ch_embed_size, embed_size, kernal_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(ch_embed_size, embed_size, kernal_size)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):                # (bch, M_ch, w_len)
        x = self.conv(x)                 # (bch, M_w, w_len-k+1)
        return self.pool(x).squeeze(-1)  # (bch, M_w)

### END YOUR CODE

