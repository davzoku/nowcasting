import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class CNN_1D(nn.Module):
    def __init__(self, n_features, lookback, layers="single"):
        super().__init__()
        self.layers = layers

        if self.layers == "single":
            self.conv1 = nn.Conv1d(
                in_channels=n_features,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            )
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.fc1 = nn.Linear(64 * (lookback // 2), 256)
            self.fc2 = nn.Linear(256, 1)
        else:
            self.conv1 = nn.Conv1d(
                in_channels=n_features,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1,
            )
            self.conv2 = nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1
            )
            self.conv3 = nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1
            )
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.fc1 = nn.Linear(256 * (lookback // 8), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.n_features = n_features
        self.lookback = lookback
        self.device = "cpu"

    def forward(self, x):
        if self.layers == "single":
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(-1, 64 * (self.lookback // 2))
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(-1, 256 * (self.lookback // 8))
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
        return x

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs,
        criterion,
        optimizer,
        min_delta=1e-4,
        patience=5,
    ):
        self.train()
        best_val_loss = np.Inf
        counter = 0
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(num_epochs)):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.reshape(-1, self.n_features, self.lookback)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.forward(x_batch)

                loss = criterion(outputs, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}", end=" ")

            train_losses.append(loss.item())

            # Compute validation loss
            with torch.no_grad():
                val_loss = 0.0
                for x_batch, y_batch in val_loader:
                    # Forward pass and loss computation
                    x_batch = x_batch.reshape(-1, self.n_features, self.lookback)
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    val_loss += criterion(self.forward(x_batch), y_batch)
                val_loss /= len(val_loader)

                print(f"Val loss:{val_loss:.4f}")
                val_losses.append(val_loss.item())

                # Check for improvement
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        # Plot the training loss
        plt.plot(range(1, epoch + 2), train_losses, label="Training Loss")
        # plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss")
        plt.show()

    def evaluate(self, test_loader, criterion, scaler):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            total_mse = 0
            total_rmse = 0
            total_mae = 0
            num_samples = 0

            for x_batch, y_batch in test_loader:
                x_batch = x_batch.reshape(-1, self.n_features, self.lookback)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.forward(x_batch)

                # Compute loss
                loss = criterion(y_pred, y_batch)

                # Compute MSE, RMSE and MAE in the original units
                y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1))
                y_batch_orig = scaler.inverse_transform(y_batch.reshape(-1, 1))
                mse = mean_squared_error(y_batch_orig, y_pred_orig)
                rmse = mean_squared_error(y_batch_orig, y_pred_orig, squared=False)
                mae = mean_absolute_error(y_batch_orig, y_pred_orig)

                # Accumulate loss, RMSE, and MAE
                total_loss += loss.item() * x_batch.size(0)
                total_mse += mse * x_batch.size(0)
                total_rmse += rmse * x_batch.size(0)
                total_mae += mae * x_batch.size(0)
                num_samples += x_batch.size(0)

            # Compute average loss, RMSE, and MAE
            avg_loss = total_loss / num_samples
            avg_mse = total_mse / num_samples
            avg_rmse = total_rmse / num_samples
            avg_mae = total_mae / num_samples

            print(
                f"Test Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}"
            )

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
            device = torch.device("cpu")
        print(f"torch.device => {device}")
        return device
