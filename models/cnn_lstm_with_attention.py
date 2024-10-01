import torch
from torch import nn
from cnn_lstm import CLSTMModel
from attention import SelfAttention
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class CLSTMAttModel(CLSTMModel):
    """
    Implementation of proposed architecture with LSTM and CNN modules with self attention.
    """

    def __init__(self, input_dim, hidden_dim=32, layer_dim=3, output_dim=7, heads=1):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.attention = SelfAttention(hidden_dim * 2).to(device)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, output_dim),
        )
        self.init_weights()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.layer_dim * 2, x.size(0), self.hidden_dim
        ).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(
            self.layer_dim * 2, x.size(0), self.hidden_dim
        ).requires_grad_()

        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        y = self.pool(x.transpose(1, 2))
        x = self.batch_norm(y)
        out, (h1, c1) = self.lstm(x, (h0.to(device), c0.to(device)))
        out = self.attention(out)
        out = self.dense(out)
        return out
