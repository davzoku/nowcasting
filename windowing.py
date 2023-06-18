import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# class WindowGenerator:
#     def __init__(self, data, lookback, lookahead, batch_size, label_idx):
#         self.lookback = lookback
#         self.lookahead = lookahead
#         self.batch_size = batch_size
#         self.data = data
#         self.label_idx = label_idx
#         self.X = []
#         self.y = []

#     def split_window(self):
#         # <-- lookback --> <----- i -----> <-- lookahead -->
#         # |---------------- data length -------------------|

#         # start at lookback, end before lookahead
#         idx_start = self.lookback
#         idx_end = len(self.data) - self.lookahead

#         for i in range(idx_start, idx_end):
#             # append to X the last n rows for lookback,
#             # and all columns as features, including the
#             # ts data (label) itself
#             self.X.append(self.data[i - self.lookback : i, :])

#             # append to y the lookahead row we want to predict,
#             # and only the label column
#             self.y.append(self.data[i + self.lookahead, self.label_idx])

#         return (
#             torch.Tensor(np.array(self.X)),
#             torch.Tensor(np.array(self.y).reshape(-1, 1)),
#         )

#     def make_dataloader(self):
#         X, y = self.split_window()
#         dataset = TSDataset(X, y)

#         # dataloader = DataLoader(dataset,
#         #                         batch_size=self.batch_size,
#         #                         shuffle=True)
#         dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
#         return dataloader


# class TSDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         features = self.X[idx]
#         label = self.y[idx]

#         return features, label

# xxxxxxx

class WindowGenerator:
    def __init__(
        self, dataset, target, lookback=6, lookahead=1, batch_size=18, multi_step=False
    ):
        self.target = target.ravel().astype(np.float32)
        self.dataset = np.column_stack((dataset, target)).astype(np.float32)
        
        self.lookback = lookback
        self.lookahead = lookahead
        self.batch_size = batch_size
        self.multi_step = multi_step
        self.input = []
        self.output = []

       
    def make_dataloader(self):
        # <-- lookback --> <----- i -----> <-- lookahead -->
        # |---------------- data length -------------------|

        # start at lookback, end before lookahead        
        X, y = self.split_data()
        dataset = TimeSeriesDataset(X, y)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    
    def split_data(self):
        for i in range(len(self.dataset) - self.lookahead - self.lookback):
            self.input.append(self.dataset[i : i + self.lookback])
            if self.multi_step == True:
                self.output.append(self.target[i + self.lookback : i + self.lookback + self.lookahead])
            else:
                self.output.append(self.target[i + self.lookback + self.lookahead])

        # if self.batch_size:
        #     self.input, self.output = self._batch_generator(self.input, self.output, self.batch_size)
        #     return self.input, self.output
        return np.array(self.input), np.array(self.output)

    # def _batch_generator(self, X, y, batch_size) -> tuple:
    #     # doesn't split if batch size is greater than the size
    #     if batch_size is None or batch_size >= X.shape[0]:
    #         return [X], [y]
    #     else:
    #         # using divmod to find the sections and remainder
    #         sections, remainder = divmod(X.shape[0], batch_size)
    #         # creates a tuple of list of arrays
    #         # the last array is the remainder which is appeneded to the list
    #         input, output = (
    #             np.split(X[:-remainder], sections),
    #             np.split(y[:-remainder], sections),
    #         )
    #         input.append(X[-remainder:])
    #         output.append(y[-remainder:])
    #         # returns a tuple of list of arrays
    #         return input, output


        
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# import numpy as np


# class WindowGenerator:
#     def __init__(
#         self, dataset, target, test_size=0.2, lookback=6, lookahead=1, batch_size=32, multi_step=False
#     ):
#         self.target = target.ravel().astype(np.float32)
#         self.dataset = np.column_stack((dataset, target)).astype(np.float32)
        
#         self.lookback = lookback
#         self.lookahead = lookahead
#         self.batch_size = batch_size
#         self.test_size = test_size
#         self.multi_step = multi_step
#         self.input = []
#         self.output = []

#     def split_data(self):
#         for i in range(len(self.dataset) - self.lookahead - self.lookback):
#             self.input.append(self.dataset[i : i + self.lookback])
#             if self.multi_step == True:
#                 self.output.append(self.target[i + self.lookback : i + self.lookback + self.lookahead])
#             else:
#                 self.output.append(self.target[i + self.lookback + self.lookahead])

#         if self.batch_size:
#             self.input, self.output = self._batch_generator(self.input, self.output, self.batch_size)
#             return self.input, self.output
#         return np.array(self.input), np.array(self.output)

#         # Initialy used train test split here
#         # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 42)
        

#     def _batch_generator(self, X, y, batch_size) -> tuple:
#         """
#         Generate batches of data
#         If batch size return the data back packaged in a list

#         Parameters
#         ----------
#         X : np.ndarray
#             Dataset
#         y : np.ndarray
#             Target array of labels
#         batch_size : int, optional
#             Size of batch, by default None

#         Returns
#         -------
#         tuple
#             Tuple of list of arrays in the form of inputs, outputs
#         """
#         # doesn't split if batch size is greater than the size
#         if batch_size is None or batch_size >= X.shape[0]:
#             return [X], [y]
#         else:
#             # using divmod to find the sections and remainder
#             sections, remainder = divmod(X.shape[0], batch_size)
#             # creates a tuple of list of arrays
#             # the last array is the remainder which is appeneded to the list
#             input, output = (
#                 np.split(X[:-remainder], sections),
#                 np.split(y[:-remainder], sections),
#             )
#             input.append(X[-remainder:])
#             output.append(y[-remainder:])
#             # returns a tuple of list of arrays
#             return input, output
