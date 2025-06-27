import torch
import torch.nn as nn
import models.attention as attention
import utils.utils as utils

"""
class LSTMCell(nn.Module) :
    def __init__(self,input_dim,output_dim) :
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim        #cell dimention is same as output / hidden state
        self.Fi = nn.Linear(input_dim,output_dim)
        self.Fh = nn.Linear(output_dim,output_dim)
        self.Ii = nn.Linear(input_dim,output_dim)
        self.Ih = nn.Linear(output_dim,output_dim)
        self.Ci = nn.Linear(input_dim,output_dim)
        self.Ch = nn.Linear(output_dim,output_dim)
        self.Oi = nn.Linear(input_dim,output_dim)
        self.Oh = nn.Linear(output_dim,output_dim)
    
    def forward(self,input,hidden_state=None,cell_state=None) :
        B,_ = input.shape
       # print('input shape:', input.shape)
        if hidden_state is None or cell_state is None:
            hidden_state = input.new_zeros(B, self.outdim)
            cell_state = input.new_zeros(B, self.outdim)

        i = torch.sigmoid(self.Ii(input) + self.Ih(hidden_state))
        f = torch.sigmoid(self.Fi(input) + self.Fh(hidden_state))
        g = torch.tanh(self.Ci(input) + self.Ch(hidden_state))
        cell_state = f * cell_state + i * g
        o = torch.sigmoid(self.Oi(input) + self.Oh(hidden_state)) 
        hidden_state = o * torch.tanh(cell_state)
        return hidden_state,cell_state
    

class StackedLSTMcell(nn.Module) :
    def __init__(self,input_dim,output_dim) :
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim 
        self.lstm1 = LSTMCell(input_dim,output_dim)
        self.lstm2 = LSTMCell(output_dim,output_dim)
    
    def forward(self,input,h1=None,c1=None,h2=None,c2=None) :
        
        h1,c1 = self.lstm1(input,h1,c1)
        h2,c2 = self.lstm2(h1,h2,c2)
        
        return h1,c1,h2,c2



class Decoder(nn.Module) :
    def __init__(self,input_dim,output_dim,vocab_size) :
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim 
        self.vocab_size = vocab_size
        self.embd = nn.Embedding(vocab_size+2,output_dim // 2)
        self.attention = attention.Attention(output_dim,output_dim // 2)
        print('output_dim:', output_dim, 'vocab_size:', vocab_size)
        self.rnn = StackedLSTMcell((output_dim // 2) * 2,output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, vocab_size * 2),
            nn.LayerNorm(vocab_size * 2),   
            nn.ReLU(),
            nn.Linear(vocab_size * 2, vocab_size)
        )

    def forward(self, input, prev_output=0, h1=None, c1=None, h2=None, c2=None,
                attn=None, target=None, pad_length=64):
    """ """
        input: (B, T_enc, C) - encoder output
        prev_output: int or (B,) tensor - initial decoder input token(s)
        target: (B, T_tgt) - target token ids (padded with PAD_ID)
        pad_length: int - max number of decoder steps
        
        """"""
        B, T_enc, _ = input.shape

        if attn is None:
            attn = torch.zeros(B, self.outdim // 2, device=input.device)

        logits_all = []
        loss = 0.0

        # Define pad id dynamically
        PAD_ID = self.vocab_size + 1
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        # Prepare initial decoder input
        if isinstance(prev_output, int):
            prev_output = torch.full((B,), prev_output, dtype=torch.long, device=input.device)
        else:
            prev_output = prev_output.to(input.device)

        # Decide number of decoding steps
        if target is not None:
            T_dec = min(target.shape[1], pad_length)

        for t in range(T_dec):
            pout = self.embd(prev_output)          # (B, D/2)
            ci = torch.cat([pout, attn], dim=-1)   # (B, D)

            h1, c1, h2, c2 = self.rnn(ci, h1, c1, h2, c2)  # RNN step
            attn = self.attention(input, h2)              # Attention step

            logits = self.mlp(h2)                         # (B, vocab_size)
            logits_all.append(logits)

            if target is not None:
                # If sequence is shorter than pad_length, fill with PAD_ID
                current_target = target[:, t] if t < target.size(1) else torch.full((B,), PAD_ID, device=input.device)
                loss += criterion(logits, current_target)
                prev_output = current_target  # Teacher forcing
            else:
                probs = torch.softmax(logits, dim=-1)
                prev_output = torch.argmax(probs, dim=-1)

        logits_all = torch.stack(logits_all, dim=1)  # (B, T_dec, vocab_size)

        if target is not None:
            loss = loss / T_dec
            return logits_all, loss
        else:
            return logits_all

    def generate(self, input, max_len=100, print_live=True):
        """"""
        Generate text from encoder output with live character printing.

        Args:
            input (Tensor): (B, T, C) encoder output
            max_len (int): maximum number of steps to decode
            print_live (bool): whether to print character as soon as it is generated

        Returns:
            decoded_strings (List[str])
            output_tensor (Tensor): (B, max_len)
        """"""
        B = input.size(0)
        assert B == 1, "Live printing only supported for batch size 1"
        device = input.device

        h1 = c1 = h2 = c2 = None
        attn = torch.zeros(B, self.outdim // 2, device=device)
        prev_output = torch.full((B,), utils.SOS_ID, dtype=torch.long, device=device)

        all_outputs = []

        for _ in range(max_len):
            pout = self.embd(prev_output)
            ci = torch.cat([pout, attn], dim=-1)
            h1, c1, h2, c2 = self.rnn(ci, h1, c1, h2, c2)
            attn = self.attention(input, h2)

            logits = self.mlp(h2)
            probs = torch.softmax(logits, dim=-1)
            prev_output = torch.argmax(probs, dim=-1)  # <-- Greedy decoding

            token_id = prev_output.item()
            all_outputs.append(token_id)

            if print_live:
                if token_id == utils.EOS_ID:
                    break
                ch = utils.idx2char.get(token_id, '?')
                print(ch, end='', flush=True)

            if token_id == utils.EOS_ID:
                break

        print()  # Newline after sequence ends
        decoded = utils.decode(all_outputs)
        return [decoded], torch.tensor(all_outputs).unsqueeze(0)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout_p = dropout

        self.embd = nn.Embedding(vocab_size + 2, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)

        self.attention = attention.Attention(hidden_dim, hidden_dim // 2)

        self.rnn = nn.LSTM(input_size=(hidden_dim // 2) * 2,
                           hidden_size=hidden_dim,
                           num_layers=2,
                           batch_first=True,
                           dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, vocab_size * 2),
            nn.LayerNorm(vocab_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vocab_size * 2, vocab_size)
        )

    def forward(self, input, prev_output=0, hidden=None, attn=None, target=None, pad_length=64):
        B, T_enc, _ = input.shape

        if attn is None:
            attn = torch.zeros(B, self.hidden_dim // 2, device=input.device)

        PAD_ID = self.vocab_size + 1
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        logits_all = []
        loss = 0.0

        if isinstance(prev_output, int):
            prev_output = torch.full((B,), prev_output, dtype=torch.long, device=input.device)

        T_dec = min(target.shape[1], pad_length) if target is not None else pad_length

        h = torch.zeros(2, B, self.hidden_dim, device=input.device)
        c = torch.zeros(2, B, self.hidden_dim, device=input.device)
        hidden = (h, c)

        for t in range(T_dec):
            pout = self.embd(prev_output)
            pout = self.dropout(pout)

            ci = torch.cat([pout, attn], dim=-1).unsqueeze(1)
            output, hidden = self.rnn(ci, hidden)
            h2 = output.squeeze(1)
            h2 = self.dropout(h2)

            attn = self.attention(input, h2)
            logits = self.mlp(h2)
            logits_all.append(logits)

            if target is not None:
                current_target = target[:, t] if t < target.size(1) else torch.full((B,), PAD_ID, device=input.device)
                if not (current_target == PAD_ID).all():
                    loss += criterion(logits, current_target)   

                # Use teacher forcing 90% of the time
                use_teacher = random.random() < 0.9
                if use_teacher:
                    prev_output = current_target
                else:
                    probs = F.softmax(logits, dim=-1)
                    prev_output = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                probs = F.softmax(logits, dim=-1)
                prev_output = torch.argmax(probs, dim=-1)

        logits_all = torch.stack(logits_all, dim=1)

        if target is not None:
            loss = loss / T_dec
            return logits_all, loss
        else:
            return logits_all

    def generate(self, input, max_len=100, print_live=True):
        B = input.size(0)
        assert B == 1, "Live printing only supported for batch size 1"
        device = input.device

        h = torch.zeros(2, B, self.hidden_dim, device=device)
        c = torch.zeros(2, B, self.hidden_dim, device=device)
        hidden = (h, c)

        attn = torch.zeros(B, self.hidden_dim // 2, device=device)
        prev_output = torch.full((B,), utils.SOS_ID, dtype=torch.long, device=device)

        all_outputs = []

        for _ in range(max_len):
            pout = self.embd(prev_output)
            ci = torch.cat([pout, attn], dim=-1).unsqueeze(1)
            output, hidden = self.rnn(ci, hidden)
            h2 = output.squeeze(1)
            attn = self.attention(input, h2)
            logits = self.mlp(h2)
            probs = torch.softmax(logits, dim=-1)
            prev_output = torch.argmax(probs, dim=-1)

            token_id = prev_output.item()
            all_outputs.append(token_id)

            if print_live:
                if token_id == utils.EOS_ID:
                    break
                ch = utils.idx2phone.get(token_id, '?')
                print(ch, end='', flush=True)

            if token_id == utils.EOS_ID:
                break

        print()
        decoded = utils.decode(all_outputs)
        return [decoded], torch.tensor(all_outputs).unsqueeze(0)
