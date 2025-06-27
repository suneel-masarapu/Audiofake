import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.decoder import Decoder
from models.encoder import Encoder

class seq2seq(nn.Module):
    def __init__(self, input_dim, output_dim, vocab_size, num_layers=3):
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.encoder = Encoder(input_dim, output_dim, num_layers)
        self.decoder = Decoder(output_dim, output_dim, vocab_size)

    def forward(self, input_seq, target_seq=None):
        encoder_output = self.encoder(input_seq)
        if target_seq is None:
           decoder_output = self.decoder(encoder_output)
           return decoder_output
        decoder_output,loss = self.decoder(encoder_output, target=target_seq)
        return decoder_output, loss
    