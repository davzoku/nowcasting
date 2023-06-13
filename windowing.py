import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WindowGenerator:
    def __init__(self, data, lookback, lookahead, batch_size, label_idx):
        self.lookback = lookback
        self.lookahead = lookahead
        self.batch_size = batch_size
        self.data = data
        self.label_idx = label_idx
        self.X = []
        self.y = []

    def split_window(self):
        # <-- lookback --> <----- i -----> <-- lookahead -->
        # |---------------- data length -------------------|

        # start at lookback, end before lookahead
        idx_start = self.lookback
        idx_end = len(self.data) - self.lookahead

        for i in range(idx_start, idx_end):
            # append to X the last n rows for lookback,
            # and all columns as features, including the
            # ts data (label) itself
            self.X.append(self.data[i - self.lookback : i, :])

            # append to y the lookahead row we want to predict,
            # and only the label column
            self.y.append(self.data[i + self.lookahead, self.label_idx])

        return (
            torch.Tensor(np.array(self.X)),
            torch.Tensor(np.array(self.y).reshape(-1, 1)),
        )

    def make_dataloader(self):
        X, y = self.split_window()
        dataset = TSDataset(X, y)

        # dataloader = DataLoader(dataset,
        #                         batch_size=self.batch_size,
        #                         shuffle=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader


class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]

        return features, label
