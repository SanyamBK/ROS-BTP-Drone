import torch
import torch.nn as nn

class DroughtLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        super(DroughtLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Model
        # batch_first=True expects input shape: (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation for outputting probability/score in [0, 1]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Initialize hidden state and cell state (optional, defaults to zeros if not provided)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch, seq_len, hidden_size)
        # _ (hn, cn) are the final hidden/cell states
        out, _ = self.lstm(x) 
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] gets the hidden state at the last time step for each sequence in the batch
        out = self.fc(out[:, -1, :])
        
        # Sigmoid activation
        out = self.sigmoid(out)
        
        return out

if __name__ == '__main__':
    # Simple shape check
    model = DroughtLSTM()
    print(model)
    dummy_input = torch.randn(32, 90, 6) # (Batch, Seq, Feat)
    output = model(dummy_input)
    print("Output shape:", output.shape) # Should be (32, 1)
