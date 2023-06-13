import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class CNN_1d(nn.Module):
    def __init__(self, n_features, lookback):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features,
                               out_channels=64,
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(8 * 12, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.n_features = n_features
        self.lookback = lookback
        # self.device = self.set_device()
        self.device = 'cpu'

    def forward(self, x):
        # x = x.to(self.device)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 8 * 12)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self,
            train_loader,
            val_loader,
            num_epochs,
            criterion,
            optimizer,
            min_delta=1e-4,
            patience=5):
        self.train()
        best_val_loss = np.Inf
        counter = 0

        for epoch in tqdm(range(num_epochs)):
            for (x_batch, y_batch) in train_loader:
                x_batch = x_batch.reshape(-1, self.n_features, self.lookback)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.forward(x_batch)

                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch {}/{}, Loss: {:.4f},'.format(epoch + 1, num_epochs,
                                                      loss.item()),
                  end=" ")

            # Compute validation loss
            with torch.no_grad():
                val_loss = 0.0
                for (x_batch, y_batch) in val_loader:
                    # Forward pass and loss computation
                    x_batch = x_batch.reshape(-1, self.n_features,
                                              self.lookback)
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    val_loss += criterion(self.forward(x_batch), y_batch)
                val_loss /= len(val_loader)

                print('Val loss:{:.4f}'.format(val_loss))


                # Check for improvement
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break

    def evaluate(self, test_loader, criterion):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            for (x_batch, y_batch) in test_loader:
                # Forward pass
                # y_pred = model.forward(x_batch.transpose(1, -2))
                x_batch = x_batch.reshape(-1, self.n_features, self.lookback)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.forward(x_batch)

                # Compute loss
                # criterion = nn.MSELoss()
                loss = criterion(y_pred, y_batch)
                print(loss)
                # Accumulate loss
                total_loss += loss.item()

            # Compute average loss
            avg_loss = total_loss / len(test_loader)
            print(f"Total Loss: {total_loss}")
            print(f"len(test_loader): {len(test_loader)}")
            print(f"Test Loss: {avg_loss:.4f}")

    def predict(self, data):
        # inputs = data.to(self.device)
        data = data.reshape(-1, self.n_features, self.lookback)
        data = data.to(self.device)

        outputs = self.forward(data)

        return outputs

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device('cpu')
        print("torch.device =>", device)
        return device
