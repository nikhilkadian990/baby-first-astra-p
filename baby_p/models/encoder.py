import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                 nn.Linear(32 * 4 * 4, latent), nn.LayerNorm(latent))

    def forward(self, pixels):
        return self.net(pixels.float() / 255.0)


def noncollapse_loss(z, floor):
    """VICReg-style batch variance and off-diagonal covariance; no semantic targets."""
    centered = z - z.mean(0)
    variance = torch.relu(floor - torch.sqrt(centered.square().mean(0) + 1e-4)).mean()
    cov = centered.T @ centered / max(len(z) - 1, 1)
    off_diagonal = cov.square().sum() - cov.diagonal().square().sum()
    return variance, off_diagonal / z.shape[-1]
