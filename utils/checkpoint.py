import torch
import os
import numpy as np

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class Checkpoint:
    """
    Create checkpoint of the best performance epoch.
    """

    def __init__(self, default=0, path=""):
        self.max_score = default
        self.min_loss = np.inf
        self.path = path

    def save_best(self, model, optimizer, epoch, seed, loss, score=None):
        if score is not None:
            if score > self.max_score:
                self.max_score = score
                data = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "accuracy": score,
                    "loss": loss,
                    "seed": seed,
                }
                torch.save(data, os.path.join(self.path, f"{model.__class__.__name__}.pt"))
        else:
            if loss < self.min_loss:
                self.min_loss = loss
                data = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": loss,
                    "seed": seed,
                }
                torch.save(data, os.path.join(self.path, f"{model.__class__.__name__}.pt"))
        return self.max_score if score is not None else self.min_loss
