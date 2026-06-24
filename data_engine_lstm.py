# data_engine_lstm.py
import torch
import numpy as np

def generate_macro_sequence(samples=1000, seq_length=50, seed=None):
    """
    Simulates a macro market cycle (sine wave) + upward drift + noise.
    Includes explicit seed control to enforce deterministic pipeline verification.
    """
    if seed is not None:
        np.random.seed(seed)
        
    time = np.linspace(0, 100, samples + seq_length + 5)
    macro_price = np.sin(time) * 50 + (time * 2) + 1900 + np.random.normal(0, 5, len(time))
    
    X = np.zeros((samples, seq_length, 1)) # [Batch, Sequence Length, Features]
    y = np.zeros((samples, 1))             # [Batch, 1 output (Future Price)]
    
    for i in range(samples):
        # The input is 50 days of history
        X[i, :, 0] = macro_price[i : i + seq_length]
        # The target is the price exactly 5 days into the future
        y[i, 0] = macro_price[i + seq_length + 4] 
        
    return torch.FloatTensor(X), torch.FloatTensor(y)