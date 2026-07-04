from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        hidden = max(32, input_dim // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        hidden = max(32, input_dim // 2)
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.log_var = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mu, log_var = self.mu(h), self.log_var(h)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, log_var


class GANomaly(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        hidden = max(32, input_dim // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, input_dim))
        self.reencoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, latent_dim))
        self.discriminator = nn.Sequential(nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        z_hat = self.reencoder(reconstruction)
        return reconstruction, z, z_hat


class HybridAnomalyNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 24):
        super().__init__()
        hidden = max(48, input_dim)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))
        self.anomaly_head = nn.Sequential(nn.Linear(latent_dim, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        return self.decoder(latent), self.anomaly_head(latent).squeeze(-1)


@dataclass
class TorchTrainingResult:
    model: nn.Module
    scores: np.ndarray
    training_time: float
    inference_time: float


def train_autoencoder(model: nn.Module, x_train: np.ndarray, x_eval: np.ndarray, epochs: int = 20, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32)), batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            loss = _reconstruction_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    training_time = time.perf_counter() - start
    scores, inference_time = score_reconstruction(model, x_eval)
    return TorchTrainingResult(model, scores, training_time, inference_time)


def score_reconstruction(model: nn.Module, x_eval: np.ndarray):
    device = next(model.parameters()).device
    x_tensor = torch.tensor(x_eval, dtype=torch.float32, device=device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(x_tensor)
        reconstruction = output[0] if isinstance(output, tuple) else output
        scores = torch.mean((x_tensor - reconstruction) ** 2, dim=1).cpu().numpy()
    return scores, time.perf_counter() - start


def _reconstruction_loss(model: nn.Module, batch: torch.Tensor):
    output = model(batch)
    if isinstance(model, VariationalAutoencoder):
        reconstruction, mu, log_var = output
        mse = nn.functional.mse_loss(reconstruction, batch)
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return mse + 0.001 * kl
    if isinstance(model, GANomaly):
        reconstruction, z, z_hat = output
        return nn.functional.mse_loss(reconstruction, batch) + 0.1 * nn.functional.mse_loss(z_hat, z)
    if isinstance(output, tuple):
        reconstruction, logits = output
        return nn.functional.mse_loss(reconstruction, batch) + 0.01 * torch.mean(torch.sigmoid(logits))
    return nn.functional.mse_loss(output, batch)
