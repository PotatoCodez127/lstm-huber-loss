# model_lstm.py
import torch.nn as nn

class MacroForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(MacroForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The Memory Engine (batch_first=True makes data shape [Batch, Seq, Features])
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # The Decision Engine (Condenses 64 memory nodes into 1 price prediction)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Pass sequence through LSTM. We don't manually initialize hidden states here;
        # PyTorch defaults them to zero, which is fine for basic batched training.
        lstm_out, _ = self.lstm(x)
        
        # Slicing: We only care about the network's "thoughts" at the very LAST time step
        final_thought = lstm_out[:, -1, :]
        
        # Pass the final thought to the Linear layer for the exact price
        prediction = self.fc(final_thought)
        
        return prediction