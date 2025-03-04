import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from obsea.encoders.models import Autoencoder


def train_sparse_autoencoder(
    model: Autoencoder,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    sparsity_weight: float,
    device: torch.device,
):
    """Train an autoencoder model as a sparse autoencoder."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in dataloader:
            inputs = data.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            mse_loss = criterion(outputs, inputs)

            # Sparsity loss
            encoded = model.encoder(inputs)
            sparsity_loss = torch.mean(torch.abs(encoded))

            loss = mse_loss + sparsity_weight * sparsity_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}"
        )

    print("Training complete.")
