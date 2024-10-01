import torch
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = device,
    accuracy_fn=None,
):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for _, sample in enumerate(dataloader):
        X, y = sample.get("data").type(torch.float).to(device), sample.get("label").to(
            device
        )
        logits = model(X)
        loss = loss_function(logits.unsqueeze(2), y)

        preds = torch.softmax(logits, dim=1)
        train_acc += accuracy_fn(preds, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_acc, train_loss


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device = device,
    accuracy_fn=None,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for _, sample in enumerate(dataloader):
            X, y = sample.get("data").type(torch.float).to(device), sample.get(
                "label"
            ).to(device)
            logits = model(X)
            loss = loss_function(logits.unsqueeze(2), y)
            preds = torch.softmax(logits, dim=1)
            test_acc += accuracy_fn(preds, y)
            test_loss += loss

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_acc, test_loss
