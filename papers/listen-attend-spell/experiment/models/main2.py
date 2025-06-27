import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from las_model import Listener, Speller
from data.data import SpeechDataset, collate_fn
from utils.utils import encode as encode_phonemes, vocab
from utils.functions import CreateOnehotVariable

# Constants
INPUT_DIM = 40
HIDDEN_DIM = 256
NUM_LAYERS = 3
BATCH_SIZE = 8  # Reduced batch size for debugging
PAD_LENGTH = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
VOCAB_SIZE = len(vocab)
PAD_ID = VOCAB_SIZE  # Set PAD_ID to be equal to VOCAB_SIZE
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

print(f"Using device: {DEVICE}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"PAD_ID: {PAD_ID}")

def safe_convert_to_onehot(labels, vocab_size, pad_id, device):
    """Safely convert label indices to one-hot vectors with bounds checking"""
    batch_size, seq_len = labels.size()
    
    # Move to CPU for safer processing, then back to GPU
    labels_cpu = labels.cpu()
    
    # Check and fix out-of-bounds values
    invalid_mask = (labels_cpu < 0) | (labels_cpu > vocab_size)
    if invalid_mask.any():
        print(f"Warning: Found {invalid_mask.sum()} invalid labels")
        print(f"Label range: {labels_cpu.min()} to {labels_cpu.max()}")
        print(f"Expected range: 0 to {vocab_size}")
        
        # Replace invalid values with pad_id
        labels_cpu = torch.where(invalid_mask, pad_id, labels_cpu)
    
    # Create one-hot encoding - vocab_size + 1 to include pad token
    onehot = torch.zeros(batch_size, seq_len, vocab_size + 1)
    
    # Scatter only valid indices
    valid_indices = labels_cpu.clamp(0, vocab_size)
    onehot.scatter_(2, valid_indices.unsqueeze(2), 1)
    
    # Move back to device
    if device.type == 'cuda':
        onehot = onehot.cuda()
    
    return onehot[:, :, :vocab_size]  # Return only vocab part, exclude pad dimension

def calculate_accuracy(predictions, targets, pad_id):
    """Calculate accuracy ignoring padded tokens"""
    if not predictions:
        return 0.0
    
    # Convert predictions to class indices
    pred_indices = torch.stack([pred.argmax(dim=-1) for pred in predictions], dim=1)
    
    # Ensure same sequence length
    min_len = min(pred_indices.size(1), targets.size(1))
    pred_indices = pred_indices[:, :min_len]
    targets = targets[:, :min_len]
    
    # Create mask for non-padded tokens
    mask = (targets != pad_id)
    
    if mask.sum() == 0:
        return 0.0
    
    # Calculate accuracy only for non-padded tokens
    correct = (pred_indices == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()

# Enable CUDA error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Data loading with extensive debugging
print("Loading data...")
df = pd.read_pickle(r'C:\Users\sunee\OneDrive\Desktop\Projects\audiofake\Audiofake\papers\listen-attend-spell\experiment\data\speechocean_logmel.pkl')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print("Creating datasets...")
train_ds = SpeechDataset(train_df, encode_phonemes, pad_length=PAD_LENGTH)
val_ds = SpeechDataset(val_df, encode_phonemes, pad_length=PAD_LENGTH)

# Debug dataset
print("Debugging dataset samples...")
for i in range(min(3, len(train_ds))):
    sample_x, sample_y = train_ds[i]
    print(f"Sample {i}: x shape={sample_x.shape}, y shape={sample_y.shape}")
    print(f"Sample {i}: y min/max={sample_y.min()}/{sample_y.max()}")
    print(f"Sample {i}: unique y values={torch.unique(sample_y)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, PAD_LENGTH, PAD_ID))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, PAD_LENGTH, PAD_ID))

# Debug batch
print("Debugging batch...")
try:
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx}: x shape={x_batch.shape}, y shape={y_batch.shape}")
        print(f"Batch {batch_idx}: y min/max={y_batch.min()}/{y_batch.max()}")
        print(f"Batch {batch_idx}: unique y values={torch.unique(y_batch)}")
        if batch_idx >= 2:  # Check first 3 batches
            break
except Exception as e:
    print(f"Error in data loading: {e}")
    sys.exit(1)

# Model setup
print("Setting up models...")
listener = Listener(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, "LSTM", USE_GPU, dropout_rate=0.1)
speller = Speller(output_class_dim=VOCAB_SIZE, speller_hidden_dim=HIDDEN_DIM, rnn_unit="LSTM",
                  speller_rnn_layer=2, use_gpu=USE_GPU, max_label_len=PAD_LENGTH,
                  use_mlp_in_attention=True, mlp_dim_in_attention=128,
                  mlp_activate_in_attention="tanh", listener_hidden_dim=HIDDEN_DIM,
                  multi_head=1, decode_mode=1)

# Move models to device
if USE_GPU:
    listener = listener.cuda()
    speller = speller.cuda()

params = list(listener.parameters()) + list(speller.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE)
criterion = nn.NLLLoss(ignore_index=PAD_ID)

# Training loop
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    listener.train()
    speller.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
        try:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            # Debug print for first batch
            if batch_idx == 0 and epoch == 1:
                print(f"\nFirst batch debug:")
                print(f"x_batch shape: {x_batch.shape}")
                print(f"y_batch shape: {y_batch.shape}")
                print(f"y_batch range: {y_batch.min()} to {y_batch.max()}")
            
            # Convert labels to one-hot for speller input
            y_onehot = safe_convert_to_onehot(y_batch, VOCAB_SIZE, PAD_ID, DEVICE)
            
            if batch_idx == 0 and epoch == 1:
                print(f"y_onehot shape: {y_onehot.shape}")
            
            optimizer.zero_grad()
            
            # Forward pass
            listener_features = listener(x_batch)
            if batch_idx == 0 and epoch == 1:
                print(f"listener_features shape: {listener_features.shape}")
            
            preds, attns = speller(listener_features, ground_truth=y_onehot, teacher_force_rate=0.9)
            
            if batch_idx == 0 and epoch == 1:
                print(f"Number of predictions: {len(preds)}")
                if preds:
                    print(f"First prediction shape: {preds[0].shape}")
            
            # Calculate loss
            T = min(len(preds), y_batch.size(1))
            
            if T > 0:
                # Stack predictions and flatten
                logits_stacked = torch.stack(preds[:T], dim=1)  # [batch, T, vocab]
                logits_flat = logits_stacked.view(-1, VOCAB_SIZE)
                
                # Flatten targets
                y_flat = y_batch[:, :T].contiguous().view(-1)
                
                # Calculate loss only for non-padded tokens
                mask = (y_flat != PAD_ID)
                if mask.any():
                    loss = criterion(logits_flat[mask], y_flat[mask])
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"NaN loss detected in batch {batch_idx}")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    accuracy = calculate_accuracy(preds[:T], y_batch[:, :T], PAD_ID)
                    total_accuracy += accuracy
                    num_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{accuracy:.4f}'
                    })
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"x_batch shape: {x_batch.shape if 'x_batch' in locals() else 'N/A'}")
            print(f"y_batch shape: {y_batch.shape if 'y_batch' in locals() else 'N/A'}")
            import traceback
            traceback.print_exc()
            continue

    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        print(f"\n[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}")
    else:
        print(f"\n[Epoch {epoch}] No valid batches processed")
        continue

    # Validation (simplified for debugging)
    print("Running validation...")
    listener.eval()
    speller.eval()
    val_loss = 0
    val_accuracy = 0
    val_batches = 0
    
    with torch.no_grad():
        for val_batch_idx, (x_val, y_val) in enumerate(tqdm(val_loader, desc="Validation")):
            try:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                y_val_onehot = safe_convert_to_onehot(y_val, VOCAB_SIZE, PAD_ID, DEVICE)
                
                listener_features = listener(x_val)
                preds, _ = speller(listener_features, ground_truth=y_val_onehot, teacher_force_rate=1.0)
                
                T = min(len(preds), y_val.size(1))
                if T > 0:
                    logits_stacked = torch.stack(preds[:T], dim=1)
                    logits_flat = logits_stacked.view(-1, VOCAB_SIZE)
                    y_flat = y_val[:, :T].contiguous().view(-1)
                    
                    mask = (y_flat != PAD_ID)
                    if mask.any():
                        loss = criterion(logits_flat[mask], y_flat[mask])
                        if not torch.isnan(loss):
                            val_loss += loss.item()
                            
                            accuracy = calculate_accuracy(preds[:T], y_val[:, :T], PAD_ID)
                            val_accuracy += accuracy
                            val_batches += 1
                        
            except Exception as e:
                print(f"Error in validation batch {val_batch_idx}: {e}")
                continue

    if val_batches > 0:
        avg_val_loss = val_loss / val_batches
        avg_val_accuracy = val_accuracy / val_batches
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'listener': listener.state_dict(),
                'speller': speller.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, "checkpoints/best_model.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Regular checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'listener': listener.state_dict(),
        'speller': speller.state_dict(),
        'optimizer': optimizer.state_dict()
    }, f"checkpoints/epoch_{epoch}.pt")

print("Training completed!")