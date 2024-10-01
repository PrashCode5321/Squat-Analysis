from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os

SEED = os.environ.get("SEED", 42)
device = os.environ.get("device", "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class SquatDataset(Dataset):
    """
    Create PyTorch dataset using filepaths and one-hot encoded classes in the reference CSV file.
    """

    def __init__(self, csv_file, root_dir, device, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, f"{self.labels.iloc[idx, 0]}.npy")

        data = np.load(img_name)
        data = np.array(data).astype(np.float32)
        data = torch.from_numpy(data).type(torch.float).to(device)

        label = self.labels.iloc[idx, 1:]
        label = np.array([label])
        label = label.astype("float").reshape(-1, 1)
        label = torch.from_numpy(label).type(torch.float).to(device)

        if self.transform:
            data = self.transform(data)

        sample = {"data": data, "label": label}
        return sample
