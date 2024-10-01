import torch
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")


def eval_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = device,
):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    true_labels = []
    pred_labels = []
    model.eval()
    with torch.inference_mode():
        for batch, sample in enumerate(dataloader):
            X, y = sample.get("data").to(device), sample.get("label").to(device)
            logits = model(X)
            preds = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(preds, dim=1)
            y_true = torch.argmax(y.squeeze(2), dim=1)
            true_labels.append(y_true)
            pred_labels.append(y_pred)
    true_labels = torch.concat(true_labels).cpu().detach().numpy()
    pred_labels = torch.concat(pred_labels).cpu().detach().numpy()
    return true_labels, pred_labels
