import torch
from torch import nn
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class GRUModel(nn.Module):
    """
    Implementation of previous architecture with GRU module.
    """

    def __init__(self, input_dim, hidden_dim=32, layer_dim=3, output_dim=7):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_norm = nn.BatchNorm1d(300, eps=0.001, momentum=0.99)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            layer_dim,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(4800, output_dim),
        )
        self.init_weights()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.layer_dim * 2, x.size(0), self.hidden_dim
        ).requires_grad_()

        # print(x.shape)
        x = self.batch_norm(x)
        out, h1 = self.gru(x, h0.to(device))
        out = self.dense(out)
        return out

    def init_weights(self):
        for name, param in self.gru.named_parameters():
            # do kernel initialization with Glorot Uniform distribution
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            # do recurrent initialization with Orthogonal matrix
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            # do bias initialization with zeros
            if "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.dense.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            if "bias" in name:
                nn.init.zeros_(param)
