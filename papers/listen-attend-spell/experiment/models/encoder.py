import torch
import torch.nn as nn
import torch.nn.functional as F


class pBLSTMcell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim
        self.lstm = nn.LSTM(input_dim, output_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, input):
        # input shape: (batch_size, seq_len, input_dim)
        output, _ = self.lstm(input)
        return output  # shape: (batch_size, seq_len, output_dim)