# train_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim

from data_engine_lstm import generate_macro_sequence
from model_lstm import MacroForecaster


def train_and_test():
    # Enforce global torch seed reproducibility
    torch.manual_seed(42)

    # 1. Load Data with fixed training seed to protect initialization boundaries
    X_train, y_train = generate_macro_sequence(1000, seed=42)

    # Calculate deterministic scaling factors directly from training statistics
    # to protect the pre-trade out-of-sample isolation firewall
    x_max = X_train.max()
    y_max = y_train.max()

    x_train_scaled = X_train / x_max
    y_train_scaled = y_train / y_max

    model = MacroForecaster()
    criterion = nn.HuberLoss(delta=1.0)

    # Transitioned from SGD to Adam optimization for rapid, adaptive surface convergence
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("Training Volatility-Resilient LSTM Forecaster...")
    epochs = 150
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train_scaled)
        loss = criterion(predictions, y_train_scaled)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch + 1:03d}/{epochs} | Huber Loss: {loss.item():.6f}")

    print("\n--- Out-of-Sample Firewall Validation ---")
    # Generate distinct, unseen validation data sequence paths using a separate seed
    X_val, y_val = generate_macro_sequence(3, seed=99)

    # Strict out-of-sample data isolation: normalize using TRAIN parameters only
    x_val_scaled = X_val / x_max

    model.eval()
    with torch.no_grad():
        scaled_preds = model(x_val_scaled)
        real_preds = scaled_preds * y_max

    for i in range(3):
        print(f"Scenario {i + 1}:")
        print(f"  Current Spot Price (Day 50)  : ${X_val[i, -1, 0].item():.2f}")
        print(f"  Ground Truth Target (Day 55) : ${y_val[i, 0].item():.2f}")
        print(f"  LSTM Dynamic Forecast        : ${real_preds[i, 0].item():.2f}")
        print(
            f"  Absolute Model Deviation     : ${abs(y_val[i, 0].item() - real_preds[i, 0].item()):.2f}\n"
        )


if __name__ == "__main__":
    train_and_test()
