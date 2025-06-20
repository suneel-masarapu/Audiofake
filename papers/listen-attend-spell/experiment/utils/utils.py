
# Allowed characters
valid_chars = "abcdefghijklmnopqrstuvwxyz0123456789 ,.'"
special_tokens = ['<sos>', '<eos>', '<unk>']

# Create vocabulary
vocab = special_tokens + list(valid_chars)

# Mapping
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

# Special token ids
SOS_ID = char2idx['<sos>']
EOS_ID = char2idx['<eos>']
UNK_ID = char2idx['<unk>']

def encode(s):
    s = s.lower()
    tokens = [SOS_ID]
    for ch in s:
        if ch in char2idx:
            tokens.append(char2idx[ch])
        else:
            tokens.append(UNK_ID)
    tokens.append(EOS_ID)
    return tokens

def decode(token_ids):
    chars = []
    for token in token_ids:
        if token == SOS_ID or token == EOS_ID:
            continue  # skip special tokens
        chars.append(idx2char.get(token, '?'))
    return ''.join(chars)


