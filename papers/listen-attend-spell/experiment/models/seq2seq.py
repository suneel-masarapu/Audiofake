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
        
        if torch.isnan(encoder_output).any():
            print("NaN detected in encoder output!")
            # Optionally raise an error to stop training
            raise ValueError("NaN detected in encoder output")
        
        if target_seq is None:
            decoder_output = self.decoder(encoder_output)
            return decoder_output
        
        decoder_output, loss = self.decoder(encoder_output, target=target_seq)
        return decoder_output, loss
    
    def generate(self, input_seq, max_len=100, print_live=True):
        """
        Generate output sequences from input sequences using the decoder's generate method.

        Args:
            input_seq (Tensor): Input tensor (B, T, input_dim).
            max_len (int): Maximum length of generated sequence.
            print_live (bool): Whether to print tokens live (only for batch size 1).

        Returns:
            List[List[int]]: List of generated token sequences (batch size B).
        """
        self.eval()
        with torch.no_grad():
            encoder_output = self.encoder(input_seq)
            # Assuming your decoder.generate returns (decoded_strings, tensor_of_token_ids)
            decoded_strings, generated_ids = self.decoder.generate(encoder_output, max_len=max_len, print_live=print_live)
            
        return decoded_strings, generated_ids


    