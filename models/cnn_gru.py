import torch
from torch import nn
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class CGRUModel(nn.Module):
    """
    Implementation of proposed architecture with GRU and CNN modules.
    """

    def __init__(self, input_dim, hidden_dim=32, layer_dim=3, kernel=3, output_dim=7):
        super(CGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.conv = nn.Conv1d(input_dim, hidden_dim * 2, 3)
        self.relu = nn.ReLU()
        self.pool = nn.Maxpool()
        self.batch_norm = nn.BatchNorm1d(298, eps=0.001, momentum=0.99)
        self.gru = nn.GRU(
            hidden_dim,
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
            nn.Linear(4768, output_dim),
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

        for name, param in self.conv.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            if "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.dense.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            if "bias" in name:
                nn.init.zeros_(param)
