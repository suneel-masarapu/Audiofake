import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,input_dim,output_dim) :
        super().__init__()
        self.Q = nn.Linear(input_dim,output_dim)
        self.K = nn.Linear(input_dim,output_dim)
        self.V = nn.Linear(input_dim,output_dim)    
    
    def forward(self, keys, query):
        """
        keys: (B, T, D_in)
        query: (B, D_in)
        returns: context vector (B, D_out)
        """
        B, T, _ = keys.shape

        query_proj = self.Q(query).unsqueeze(1)  # (B, 1, D_out)
        keys_proj = self.K(keys)                # (B, T, D_out)
        values_proj = self.V(keys)              # (B, T, D_out)

        scores = query_proj.bmm(keys_proj.transpose(1, 2)) / (keys_proj.size(-1) ** 0.5)  # (B, 1, T)
        attn_weights = scores.softmax(dim=-1)  # (B, 1, T)

        context = attn_weights.bmm(values_proj).squeeze(1)  # (B, D_out)
        return context
