import torch
import torch.nn as nn
import torch.nn.functional as F
import decoder.LstmCell as lstmCell



    

class pBLSTMlayer(nn.Module):  # FIXED
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim
        self.lstmf = lstmCell.LSTMCell(input_dim, output_dim // 4)
        self.lstmb = lstmCell.LSTMCell(input_dim, output_dim // 4)

    def forward(self, input):
        hf = cf = hb = cb = None
        B, T, _ = input.shape
        outputf = []
        outputb = []
        for t in range(T):
            hf, cf = self.lstmf(input[:, t, :], hf, cf)
            outputf.append(hf)
            hb, cb = self.lstmb(input[:, T - 1 - t, :], hb, cb)
            outputb.append(hb)
        # Stack time dimension
        output = torch.stack([torch.cat([outputf[t], outputb[T - 1 - t]], dim=-1) for t in range(T)], dim=1)
        return output

    def merge_neighbours(self, input):
        B, T, C = input.shape
        assert T % 2 == 0, "Time length must be even to merge neighbors"
        output = torch.zeros((B, T // 2, C * 2), device=input.device)
        for t in range(T // 2):
            output[:, t, :] = torch.cat((input[:, 2 * t, :], input[:, 2 * t + 1, :]), dim=-1)
        return output


class encoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3):
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([pBLSTMlayer(input_dim, output_dim) for _ in range(num_layers)])

    def forward(self, input):
        output = input
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i != self.num_layers - 1:
                output = layer.merge_neighbours(output)
        return output
