import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoder import LSTMCell as lstmCell



    

class pBLSTMlayer(nn.Module):  # FIXED
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim
        self.lstmf = lstmCell(input_dim, output_dim // 2)
        self.lstmb = lstmCell(input_dim, output_dim // 2)

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
        is_odd = T % 2 == 1

        if is_odd:
            # Repeat the last frame to make T even
            last_frame = input[:, -1:, :]  # shape (B, 1, C)
            input = torch.cat([input, last_frame], dim=1)
            T += 1  # Now T is even

        output = torch.zeros((B, T // 2, C * 2), device=input.device)
        for t in range(T // 2):
            output[:, t, :] = torch.cat((input[:, 2 * t, :], input[:, 2 * t + 1, :]), dim=-1)

        return output



class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3):
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim
        self.num_layers = num_layers

        layers = []
        current_input_dim = input_dim
        for i in range(num_layers):
            layer = pBLSTMlayer(current_input_dim, output_dim)
            layers.append(layer)
            # After merge_neighbours, feature dim becomes 2 * output_dim
            current_input_dim = output_dim * 2

        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        output = input
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i != self.num_layers - 1:
                output = layer.merge_neighbours(output)
        return output
