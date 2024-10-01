import torch
from torch import nn
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class SelfAttention(nn.Module):
    """
    Implemntation of Luong Attention Mechanism.
    """

    def __init__(self, units=64, bias=False):
        super(SelfAttention, self).__init__()
        self.units = units
        self.bias = bias

        self.luong_w = nn.Linear(units, units, bias=False).to(device)
        self.softmax_normalizer = nn.Softmax(dim=-1)
        self.w_c = nn.Linear(units * 2, units, bias=False).to(device)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, inputs):
        h_s = inputs.to(device)
        batch_size, seq_len, units = h_s.size()
        h_t = h_s[:, -1, :]  # Last hidden state
        s = self.luong_w(h_s)
        score = torch.bmm(h_t.unsqueeze(1), s.transpose(1, 2)).squeeze(1)
        alpha_s = self.softmax_normalizer(score)
        context_vector = torch.sum(h_s * alpha_s.unsqueeze(-1), dim=1)
        concat = torch.cat([context_vector, h_t], dim=-1)
        a_t = self.tanh(self.w_c(concat))
        return a_t

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            if self.bias:
                if "bias" in name:
                    nn.init.zeros_(param)
