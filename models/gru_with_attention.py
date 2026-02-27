import torch
from torch import nn
from models.gru import GRUModel
from models.attention import SelfAttention
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class GRUAttModel(GRUModel):
    """
    Implementation of previous architecture with GRU module with self attention.
    """

    def __init__(self, input_dim, hidden_dim=32, layer_dim=3, output_dim=7, heads=1):
        super().__init__(input_dim, hidden_dim, layer_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # self.attention = nn.MultiheadAttention(hidden_dim*2, heads, batch_first=True)
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

        x = self.batch_norm(x)
        out, h1 = self.gru(x, h0.to(device))
        out = self.attention(out)
        out = self.dense(out)
        return out
