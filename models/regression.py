import torch
from torch import nn
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class RegressionModel(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=12):
        super(RegressionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.3),  # nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 3),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LeakyReLU(0.3),  # nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.init_weights()

    def forward(self, x):
        out = self.dense(x)
        return out

    def init_weights(self):
        for name, param in self.dense.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            if "bias" in name:
                nn.init.zeros_(param)
