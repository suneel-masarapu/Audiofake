import torch
import torch.nn as nn
import models.attention as attention
import utils.utils as utils

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



class decoder(nn.Module) :
    def __init__(self,input_dim,output_dim,vocab_size) :
        super().__init__()
        self.indim = input_dim
        self.outdim = output_dim 
        self.embd = nn.Embedding(vocab_size,output_dim // 2)
        self.attention = attention.Attention(output_dim,output_dim // 2)
        print('output_dim:', output_dim, 'vocab_size:', vocab_size)
        self.rnn = StackedLSTMcell((output_dim // 2) * 2,output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, vocab_size * 2),
            nn.LayerNorm(vocab_size * 2),   
            nn.ReLU(),
            nn.Linear(vocab_size * 2, vocab_size)
        )

    def forward(self, input, prev_output=0, h1=None, c1=None, h2=None, c2=None, attn=None, target=None):
        B, T, C = input.shape

        if attn is None:
            attn = torch.zeros(B, self.outdim // 2, device=input.device)   #(B, output_dim // 2)

        logits_all = []
        loss = 0
        criterion = nn.CrossEntropyLoss()

        # Make sure prev_output is a tensor
        if isinstance(prev_output, int):
            prev_output = torch.full((B,), prev_output, dtype=torch.long, device=input.device) #(B,)
        else:
            prev_output = prev_output.to(input.device)

        for t in range(T):
            pout = self.embd(prev_output)  # (B, output_dim // 2)
            ci = torch.cat([pout, attn], dim=-1)  # (B, output_dim)

            h1, c1, h2, c2 = self.rnn(ci, h1, c1, h2, c2) # (B, output_dim), (B, output_dim), (B, output_dim), (B, output_dim)
            #print('input shape:', input.shape, 'h2 shape:', h2.shape, 'attn shape:', attn.shape)
            attn = self.attention(input, h2)

            logits = self.mlp(h2)  # (B, vocab_size)
            logits_all.append(logits)

            probs = torch.softmax(logits, dim=-1)
            prev_output = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape (B,)

            if target is not None:
                loss += criterion(logits, target[:, t])

        logits_all = torch.stack(logits_all, dim=1)  # (B, T, vocab_size)

        if target is not None:
            return logits_all, loss
        else:
            return logits_all
        

    def generate(self, input, max_len=100, print_live=True):
        """
        Generate text from encoder output with live character printing.

        Args:
            input (Tensor): (B, T, C) encoder output
            max_len (int): maximum number of steps to decode
            print_live (bool): whether to print character as soon as it is generated

        Returns:
            decoded_strings (List[str])
            output_tensor (Tensor): (B, max_len)
        """
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
            prev_output = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

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

            






