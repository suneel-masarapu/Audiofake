import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import torch.optim as optim
import os
import sys

# Add the parent directory (i.e., experiment/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.decoder import Decoder
from models.encoder import Encoder
from models.seq2seq import seq2seq
from utils.utils import vocab
from data.data import SpeechDataset, collate_fn
from utils.utils import encode as encode_phonemes

# Set the random seed for reproducibility
torch.manual_seed(42)

# Load data
df = pd.read_pickle(r'C:\Users\sunee\OneDrive\Desktop\Projects\audiofake\Audiofake\papers\listen-attend-spell\experiment\data\speechocean_logmel.pkl')

# Constants
vocab_size = len(vocab)
PAD_ID = vocab_size + 1
BATCH_SIZE = 16
PAD_LENGTH = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# Train/Val split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_ds = SpeechDataset(train_df, encode_phonemes, pad_length=PAD_LENGTH)
val_ds = SpeechDataset(val_df, encode_phonemes, pad_length=PAD_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, PAD_LENGTH, PAD_ID))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, PAD_LENGTH, PAD_ID))

model = seq2seq(input_dim=40, output_dim=256, vocab_size=vocab_size, num_layers=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Training on {DEVICE}")

model.eval()
val_loss = 0
with torch.no_grad():
    for x_val, y_val in val_loader:
        x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
        _, loss = model(x_val, target_seq=y_val)
        val_loss += loss.item()

avg_val_loss = val_loss / len(val_loader)
print(f"[Epoch 0] Val Loss:   {avg_val_loss:.4f}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for step, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        if torch.isnan(x_batch).any():
            print(f"NaN in input at epoch {epoch}, step {step}")
            exit(1)
        if torch.isnan(y_batch).any():
            print(f"NaN in target at epoch {epoch}, step {step}")
            exit(1)

        optimizer.zero_grad()
        outputs, loss = model(x_batch, target_seq=y_batch)

        if torch.isnan(outputs).any() or torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}, step {step}")
            exit(1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
            _, loss = model(x_val, target_seq=y_val)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"[Epoch {epoch}] Val Loss:   {avg_val_loss:.4f}")

    # Save model checkpoint
    checkpoint_path = f"checkpoints/epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
