"""Training functions for the encoders module."""

# pylint: disable=too-many-arguments

from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from obsea import config
from obsea.encoders.models import Autoencoder
from obsea.utils import console


def train_sparse_autoencoder(
    model: Autoencoder,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    sparsity_weight: float = 1e-5,
    early_stopping: Optional[int] = None,
):
    """Train an autoencoder model as a sparse autoencoder."""
    model.to(config.device)  # Move the model to the device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        console.print(f"[bold green]Epoch {epoch + 1}/{epochs}[/bold green]")
        train_loss = _train_epoch(model, train_dataloader, criterion, optimizer, sparsity_weight)
        _log_epoch(epoch, epochs, train_loss, train_dataloader, "Training")

        val_loss = _validate_epoch(model, val_dataloader, criterion, sparsity_weight)
        _log_epoch(epoch, epochs, val_loss, val_dataloader, "Validation")

        if early_stopping:
            if _check_early_stopping(val_loss, best_val_loss, patience_counter, early_stopping):
                console.print("[bold red]Early stopping triggered.[/bold red]")
                break
            best_val_loss, patience_counter = _update_early_stopping(val_loss, best_val_loss, patience_counter)

    console.print("[bold blue]Training complete.[/bold blue]")


def _train_epoch(model, dataloader, criterion, optimizer, sparsity_weight):
    model.train()
    running_loss = 0.0
    for data in tqdm(dataloader, desc="Training", leave=False):
        inputs = data.to(config.device)
        loss = _train_step(model, inputs, criterion, optimizer, sparsity_weight)
        running_loss += loss.item()
    return running_loss


def _validate_epoch(model, dataloader, criterion, sparsity_weight):
    model.eval()
    val_loss = 0.0
    for data in tqdm(dataloader, desc="Validation", leave=False):
        inputs = data.to(config.device)
        loss = _val_step(model, inputs, criterion, sparsity_weight)
        val_loss += loss.item()
    return val_loss


def _train_step(model, inputs, criterion, optimizer, sparsity_weight):
    optimizer.zero_grad()
    encoded, decoded = model(inputs)
    loss_mse = criterion(decoded, inputs)
    loss_sparsity = torch.norm(encoded, p=1)
    loss = loss_mse + sparsity_weight * loss_sparsity
    loss.backward()
    optimizer.step()
    return loss


def _val_step(model, inputs, criterion, sparsity_weight):
    encoded, decoded = model(inputs)
    loss_mse = criterion(decoded, inputs)
    loss_sparsity = torch.norm(encoded, p=1)
    loss = loss_mse + sparsity_weight * loss_sparsity
    return loss


def _log_epoch(epoch, epochs, running_loss, dataloader, phase):
    console.print(
        (
            f"{phase} Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {running_loss / len(dataloader):.4f}" 
        )  # fmt: skip
    )


def _check_early_stopping(val_loss, best_val_loss, patience_counter, early_stopping):
    if val_loss < best_val_loss:
        return False
    patience_counter += 1
    if patience_counter >= early_stopping:
        return True
    return False


def _update_early_stopping(val_loss, best_val_loss, patience_counter):
    if val_loss < best_val_loss:
        return val_loss, 0
    return best_val_loss, patience_counter
