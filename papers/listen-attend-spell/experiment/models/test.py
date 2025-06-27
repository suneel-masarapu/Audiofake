import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from las_model import Listener, Speller
from utils.utils import encode as encode_phonemes, vocab, decode as decode_phonemes
from utils.functions import CreateOnehotVariable

# Constants (should match your training script)
INPUT_DIM = 40
HIDDEN_DIM = 256
NUM_LAYERS = 3
PAD_LENGTH = 64
VOCAB_SIZE = len(vocab)
PAD_ID = VOCAB_SIZE
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

print(f"Using device: {DEVICE}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Vocab type: {type(vocab)}")

def decode_predictions(predictions, vocab, threshold=0.5):
    """Convert model predictions to readable text"""
    decoded_sequence = []
    
    # Check if vocab is a dictionary or list
    if isinstance(vocab, dict):
        # vocab is a dictionary mapping phoneme -> index
        vocab_items = list(vocab.items())
        idx_to_phoneme = {idx: phoneme for phoneme, idx in vocab_items}
    elif isinstance(vocab, list):
        # vocab is a list of phonemes
        idx_to_phoneme = {idx: phoneme for idx, phoneme in enumerate(vocab)}
    else:
        print(f"Unknown vocab type: {type(vocab)}")
        return []
    
    for pred in predictions:
        # Get the most likely token
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 2:  # [batch_size, vocab_size]
                token_idx = pred.argmax(dim=-1)[0].item()  # Take first in batch
            else:  # [vocab_size]
                token_idx = pred.argmax().item()
        else:
            token_idx = pred
        
        # Convert index to phoneme
        if token_idx in idx_to_phoneme:
            phoneme = idx_to_phoneme[token_idx]
            decoded_sequence.append(phoneme)
        else:
            decoded_sequence.append('<UNK>')
    
    return decoded_sequence

def test_model_inference(listener, speller, input_data, max_length=None):
    """Test the model in inference mode (no teacher forcing)"""
    listener.eval()
    speller.eval()
    
    with torch.no_grad():
        # Encode input
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(0)  # Add batch dimension
        
        input_data = input_data.to(DEVICE)
        
        # Get listener features
        listener_features = listener(input_data)
        print(f"Listener features shape: {listener_features.shape}")
        
        # Generate predictions without teacher forcing
        predictions, attention_weights = speller(
            listener_features, 
            ground_truth=None,  # No ground truth for inference
            teacher_force_rate=0.0  # No teacher forcing
        )
        
        return predictions, attention_weights, listener_features

def main():
    # Debug vocab structure
    print(f"Vocab sample: {vocab if isinstance(vocab, list) else list(vocab.items())[:5]}")
    
    # Initialize models
    print("Initializing models...")
    listener = Listener(
        input_feature_dim=INPUT_DIM,
        listener_hidden_dim=HIDDEN_DIM,
        listener_layer=NUM_LAYERS,
        rnn_unit="LSTM",
        use_gpu=USE_GPU,
        dropout_rate=0.1
    )
    
    speller = Speller(
        output_class_dim=VOCAB_SIZE,
        speller_hidden_dim=HIDDEN_DIM,
        rnn_unit="LSTM",
        speller_rnn_layer=2,
        use_gpu=USE_GPU,
        max_label_len=PAD_LENGTH,
        use_mlp_in_attention=True,
        mlp_dim_in_attention=128,
        mlp_activate_in_attention="tanh",
        listener_hidden_dim=HIDDEN_DIM,
        multi_head=1,
        decode_mode=1
    )
    
    # Move models to device
    if USE_GPU:
        listener = listener.cuda()
        speller = speller.cuda()
    
    # Load checkpoint
    checkpoint_path = r'C:\Users\sunee\OneDrive\Desktop\Projects\audiofake\Audiofake\papers\listen-attend-spell\experiment\checkpoints\best_model.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Load model states
        listener.load_state_dict(checkpoint['listener'])
        speller.load_state_dict(checkpoint['speller'])
        
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Using randomly initialized model for testing...")
    
    # Load test data
    print("Loading test data...")
    df = pd.read_pickle(r'C:\Users\sunee\OneDrive\Desktop\Projects\audiofake\Audiofake\papers\listen-attend-spell\experiment\data\speechocean_logmel.pkl')
    
    # Test on multiple examples
    num_test_samples = min(5, len(df))
    print(f"Testing on {num_test_samples} samples...")
    
    for i in range(num_test_samples):
        print(f"\n{'='*50}")
        print(f"Testing sample {i+1}/{num_test_samples}")
        print(f"{'='*50}")
        
        # Get example
        example = df.iloc[i]
        input_data = example['log_mel']
        true_phonemes = example['phonemes']
        
        print(f"Input shape: {np.array(input_data).shape}")
        print(f"True phonemes: {true_phonemes}")
        
        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        try:
            # Test inference
            predictions, attention_weights, listener_features = test_model_inference(
                listener, speller, input_tensor
            )
            
            print(f"Number of predictions: {len(predictions)}")
            
            if predictions:
                # Decode predictions
                decoded_phonemes = decode_predictions(predictions, vocab)
                predicted_text = ' '.join(decoded_phonemes)
                
                print(f"Predicted phonemes: {decoded_phonemes}")
                print(f"Predicted text: {predicted_text}")
                
                # Show attention info
                if attention_weights:
                    print(f"Attention weights shape: {[att.shape for att in attention_weights[:3]]}")
                
                # Calculate some basic metrics
                pred_length = len(decoded_phonemes)
                true_length = len(true_phonemes.split()) if isinstance(true_phonemes, str) else len(true_phonemes)
                print(f"Prediction length: {pred_length}, True length: {true_length}")
                
                # Show first few predictions vs ground truth
                print(f"First 10 predicted: {decoded_phonemes[:10]}")
                if isinstance(true_phonemes, str):
                    true_list = true_phonemes.split()
                else:
                    true_list = true_phonemes
                print(f"First 10 true: {true_list[:10]}")
                
            else:
                print("No predictions generated")
                
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with different decode modes
    print(f"\n{'='*50}")
    print("Testing different decode modes")
    print(f"{'='*50}")
    
    example = df.iloc[0]
    input_tensor = torch.tensor(example['log_mel'], dtype=torch.float32)
    
    for decode_mode in [0, 1, 2]:
        print(f"\nTesting decode mode {decode_mode}:")
        
        # Update speller decode mode
        speller.decode_mode = decode_mode
        
        try:
            predictions, _, _ = test_model_inference(listener, speller, input_tensor)
            
            if predictions:
                decoded_phonemes = decode_predictions(predictions, vocab)
                print(f"Mode {decode_mode} result: {' '.join(decoded_phonemes[:10])}...")  # Show first 10
            else:
                print(f"Mode {decode_mode}: No predictions")
                
        except Exception as e:
            print(f"Mode {decode_mode} error: {e}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()