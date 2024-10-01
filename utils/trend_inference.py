from tqdm.auto import tqdm
import torch
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def inference(model, angles_smooth):
    w = 18
    output = []
    model.eval()
    with torch.inference_mode():
        for i in range(angles_smooth.shape[0]):
            if w + i + 1 > angles_smooth.shape[0]:
                break
            X, y = angles_smooth[i : w + i], angles_smooth[1 + i : w + i + 1]
            logits = model(X)
            output.append(logits)
    combined = torch.cat([torch.cat(output), angles_smooth[:18, :]])
    # print(combined.shape, angles_smooth.shape, torch.cat(output).reshape(-1, 18, 12).shape)
    c = torch.cat(output).reshape(-1, 18, 12)
    c = torch.cat([angles_smooth[0, :].unsqueeze(0), c[0, :, :], c[1:, -1, :]])
    return c
