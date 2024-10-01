from ignite.engine import *
from ignite.metrics import *
import torch
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def process_function(engine, batch):
    y_pred, y = batch
    return y_pred, y


def one_hot_to_binary_output_transform(output):
    y_pred, y = output
    y = torch.argmax(y, dim=1)  # one-hot vector to label index vector
    return y_pred, y


class CategoricalAccuracy:
    """
    Categorical Accuracy function using PyTorch Ignite.
    """

    def __init__(self, k=1) -> None:
        self.engine = Engine(process_function)
        self.metric = TopKCategoricalAccuracy(
            k=k, output_transform=one_hot_to_binary_output_transform
        )
        self.metric.attach(self.engine, "top_k_accuracy")

    def cat_acc(self, prediction, true):
        state = self.engine.run([[prediction, true.squeeze(-1)]])
        acc = state.metrics["top_k_accuracy"]
        return acc
