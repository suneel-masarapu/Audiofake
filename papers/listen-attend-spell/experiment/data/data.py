import torch
from torch.utils.data import Dataset



class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, pad_length=64):
        self.df = df
        self.tokenizer = tokenizer
        self.pad_length = pad_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        log_mel = torch.tensor(row["log_mel"], dtype=torch.float32)
        tokens = self.tokenizer(row["phonemes"])  # [<sos>, ..., <eos>]
        return log_mel, torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch, pad_length=64, pad_id=73):
    mels, targets = zip(*batch)

    # Let input T_enc be dynamic per batch
    T_enc_max = max(m.shape[0] for m in mels)
    feat_dim = mels[0].shape[1]
    padded_mels = torch.zeros(len(mels), T_enc_max, feat_dim)

    padded_targets = torch.full((len(targets), pad_length), pad_id, dtype=torch.long)

    for i, (mel, tgt) in enumerate(zip(mels, targets)):
        padded_mels[i, :mel.shape[0]] = mel
        tgt_len = min(len(tgt), pad_length)
        padded_targets[i, :tgt_len] = tgt[:tgt_len]

    return padded_mels, padded_targets
