#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):

    def __init__(self, embed_size, drop_rate):
        super(Highway, self).__init__()
        self.proj = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x_proj = F.relu(self.proj(x))                    # (bch, M_w)
        x_gate = torch.sigmoid(self.gate(x))             # (bch, M_w)
        x_highway = x_gate * x_proj + (1 - x_gate) * x   # (bch, M_w)
        return self.dropout(x_highway)

### END YOUR CODE

