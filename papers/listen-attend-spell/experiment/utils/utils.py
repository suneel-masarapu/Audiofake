# 🗣️ Full ARPAbet phoneme set (with stress markers)
arpabet_phones = [
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2",
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R",
    "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"
]

# 🧩 Special tokens
special_tokens = ['<sos>', '<eos>', '<unk>']

# 📚 Full vocabulary
vocab = special_tokens + arpabet_phones
vocab_size = len(vocab)

# 🔁 Index mappings
phone2idx = {phone: idx for idx, phone in enumerate(vocab)}
idx2phone = {idx: phone for phone, idx in phone2idx.items()}

# 🔑 Special token IDs
SOS_ID = phone2idx['<sos>']
EOS_ID = phone2idx['<eos>']
UNK_ID = phone2idx['<unk>']

# 🔐 Encoding: list of phoneme strings → list of token IDs
def encode(phoneme_list):
    tokens = [SOS_ID]
    for ph in phoneme_list:
        tokens.append(phone2idx.get(ph, UNK_ID))
    tokens.append(EOS_ID)
    return tokens

# 🔓 Decoding: list of token IDs → list of phoneme strings
def decode(token_ids):
    phones = []
    for token in token_ids:
        if token in [SOS_ID, EOS_ID]:
            continue
        phones.append(idx2phone.get(token, '<unk>'))
    return phones

# 🧪 Example
if __name__ == "__main__":
    sample_phonemes = ['HH', 'AH0', 'L', 'OW1']
    encoded = encode(sample_phonemes)
    print("Encoded:", encoded)

    decoded = decode(encoded)
    print("Decoded:", decoded)
