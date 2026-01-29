import torch
from torchvision.utils import save_image
from model import VAE
from preprocess import train_loader  # To get real samples
import os

# Settings
LATENT_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/vae_epoch120.pth"
OUTPUT_PATH = os.path.join("keep", "interpolation.png")
STEPS = 10  # Number of interpolation frames

# Load model
vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
vae.eval()

# Get two real images from dataset
data_iter = iter(train_loader)
batch = next(data_iter)
if isinstance(batch, (list, tuple)):
    img_batch = batch[0]  # just images, ignore labels if present
else:
    img_batch = batch
img1 = img_batch[0].unsqueeze(0).to(DEVICE)
img2 = img_batch[1].unsqueeze(0).to(DEVICE)

# Encode to latent space
with torch.no_grad():
    mu1, _ = vae.encoder(img1)
    mu2, _ = vae.encoder(img2)

    # Interpolation
    interpolated_images = []
    for alpha in torch.linspace(0, 1, steps=STEPS):
        z = mu1 * (1 - alpha) + mu2 * alpha
        recon = vae.decoder(z)
        interpolated_images.append(recon.squeeze(0))

    # Stack and save as grid
    grid = torch.stack(interpolated_images)
    save_image(grid, OUTPUT_PATH, nrow=STEPS)

print(f" Interpolation saved to {OUTPUT_PATH}")
