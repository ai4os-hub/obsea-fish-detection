import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from obsea import config
from obsea.encoders.models import Autoencoder
from obsea.utils import console


def train_sparse_autoencoder(
    model: Autoencoder,
    dataloader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    sparsity_weight: float = 1e-5,
):
    """Train an autoencoder model as a sparse autoencoder."""
    model.to(config.device)  # Move the model to the device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in dataloader:
            inputs = data.to(config.device)
            loss = _train_step(
                model, inputs, criterion, optimizer, sparsity_weight
            )
            running_loss += loss.item()

        _log_epoch(epoch, epochs, running_loss, dataloader)

    console.print("[bold blue]Training complete.[/bold blue]")


def _train_step(model, inputs, criterion, optimizer, sparsity_weight):
    optimizer.zero_grad()
    outputs = model(inputs)
    mse_loss = criterion(outputs, inputs)

    # Sparsity loss
    encoded = model.encoder(inputs)
    sparsity_loss = torch.mean(torch.abs(encoded))

    loss = mse_loss + sparsity_weight * sparsity_loss
    loss.backward()
    optimizer.step()
    return loss


def _log_epoch(epoch, epochs, running_loss, dataloader):
    console.print(
        (
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {running_loss / len(dataloader):.4f}"
        )
    )
