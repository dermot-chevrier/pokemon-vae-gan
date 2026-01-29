import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 64  # can be 64–256 depending on results
beta = 4

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 32 -> 64
            nn.Sigmoid()  # Output in [-1, 1] change to tanh for MSE and sigmoid for BCE   |   MSE/BCE
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.deconv(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    

def vae_loss_function(recon_x, x, mu, logvar,):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')  # reconstruction loss |MSE/BCE| recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return recon_loss + (beta * kl_div)