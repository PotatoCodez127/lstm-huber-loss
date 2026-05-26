# train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_engine_lstm import generate_macro_sequence
from model_lstm import MacroForecaster

def train_and_test():
    # 1. Load Data
    X, y = generate_macro_sequence(1000)
    
    # Scale data down for training stability (Neural nets hate massive numbers like 2000)
    # In production, use scikit-learn's MinMaxScaler. Here we do a crude division.
    X_scaled = X / 2500.0
    y_scaled = y / 2500.0
    
    model = MacroForecaster()
    
    # 2. The Math Components
    criterion = nn.HuberLoss(delta=1.0) # Protects against XAUUSD spikes
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # Physics-based optimizer
    
    print("Training LSTM Time Machine...")
    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_scaled)
        loss = criterion(predictions, y_scaled)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 30 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Huber Loss: {loss.item():.6f}")

    print("\n--- Live Macro Forecast Test ---")
    X_new, y_real = generate_macro_sequence(3)
    X_new_scaled = X_new / 2500.0
    
    model.eval()
    with torch.no_grad():
        scaled_preds = model(X_new_scaled)
        # Un-scale the predictions back to real Gold prices
        real_preds = scaled_preds * 2500.0
        
    for i in range(3):
        print(f"Scenario {i+1}:")
        print(f"  Current Price (Day 50): ${X_new[i, -1, 0].item():.2f}")
        print(f"  Actual Future Price (Day 55): ${y_real[i, 0].item():.2f}")
        print(f"  LSTM Prediction (Day 55): ${real_preds[i, 0].item():.2f}\n")

if __name__ == "__main__":
    train_and_test()