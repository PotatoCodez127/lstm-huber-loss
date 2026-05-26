# test_lstm.py
import torch
import numpy as np
from data_engine_lstm import generate_macro_sequence
from model_lstm import MacroForecaster

def evaluate_forecaster():
    # 1. Initialize the architecture
    model = MacroForecaster()
    
    # In a production pipeline, you would load saved weights here:
    # model.load_state_dict(torch.load("macro_forecaster.pth"))
    # For this standalone test script, we simulate inference using the network state
    model.eval() 
    
    print("--- Running LSTM Diagnostic Engine ---")
    
    # 2. Generate clean out-of-sample evaluation data (unseen by training)
    X_test, y_test = generate_macro_sequence(samples=5, seq_length=50)
    
    # Neural networks perform math efficiently on numbers bounded between 0 and 1.
    # We apply the exact same scaling factor used during the training phase.
    X_scaled = X_test / 2500.0
    
    # 3. Shape Validation (Crucial diagnostic step for 3D Tensors)
    print("\n[DATA SHAPE VERIFICATION]")
    print(f"Input Tensor Shape: {X_test.shape} -> [Batch Size, Sequence Length, Features]")
    print(f"Target Tensor Shape: {y_test.shape} -> [Batch Size, Target Coordinates]")
    
    if len(X_test.shape) == 3:
        print("✅ Input Tensor dimensions match PyTorch LSTM requirements.")
    else:
        print("❌ Critical Dimension Error: Expected a 3D Tensor.")

    # 4. Run Forward Pass Inference
    with torch.no_grad():
        scaled_predictions = model(X_scaled)
        # Reverse the mathematical scaling to get the actual projected dollar value of Gold
        real_predictions = scaled_predictions * 2500.0

    print("\n[INFERENCE VISUALIZATION DATA]")
    # 5. Extract scalar values from the tensors to calculate error distances
    for i in range(len(X_test)):
        current_price = X_test[i, -1, 0].item()
        actual_future = y_test[i, 0].item()
        predicted_future = real_predictions[i, 0].item()
        
        # Calculate absolute variance
        dollar_error = abs(actual_future - predicted_future)
        percent_error = (dollar_error / actual_future) * 100
        
        print(f"Sample {i+1}:")
        print(f"  • Current Spot Price (Day 50): ${current_price:.2f}")
        print(f"  • Ground Truth Target (Day 55): ${actual_future:.2f}")
        print(f"  • Model's Extrapolated Output : ${predicted_future:.2f}")
        print(f"  • Mathematical Variance       : ${dollar_error:.2f} ({percent_error:.2f}% Dev)")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_forecaster()