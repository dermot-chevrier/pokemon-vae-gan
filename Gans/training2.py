import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
import os

from model import Generator, Discriminator, weights_init
from data_loader import get_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
IMAGE_SIZE = 64
Z_DIM = 64
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 200
FEATURE_MAPS = 32
FLIP_LABEL_PROB = 0.05
SAMPLE_DIR = "gen_images"
os.makedirs(SAMPLE_DIR, exist_ok=True)

if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=0)

    G = Generator(z_dim=Z_DIM, img_channels=3, feature_maps=FEATURE_MAPS).to(DEVICE)
    D = Discriminator(img_channels=3, feature_maps=FEATURE_MAPS).to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerG = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))

    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=DEVICE)

    for epoch in range(1, NUM_EPOCHS + 1):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            real_labels = (0.7 + 0.3 * torch.rand(batch_size, 1, device=DEVICE)).clamp(0, 1)
            fake_labels = (0.0 + 0.3 * torch.rand(batch_size, 1, device=DEVICE)).clamp(0, 1)

            flip_real = torch.rand(batch_size) < FLIP_LABEL_PROB
            flip_fake = torch.rand(batch_size) < FLIP_LABEL_PROB
            real_labels[flip_real] = 0.0
            fake_labels[flip_fake] = 1.0

            D.zero_grad()
            outputs_real = D(real_images).view(-1, 1)
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake_images = G(noise)
            outputs_fake = D(fake_images.detach()).view(-1, 1)
            loss_D = criterion(outputs_real, real_labels) + criterion(outputs_fake, fake_labels)
            loss_D.backward()
            optimizerD.step()

            G.zero_grad()
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake_images = G(noise)
            outputs = D(fake_images).view(-1, 1)
            loss_G = criterion(outputs, torch.ones(batch_size, 1, device=DEVICE))
            loss_G.backward()
            optimizerG.step()

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")
        with torch.no_grad():
            fake_samples = G(fixed_noise).detach().cpu()
            grid = utils.make_grid(fake_samples, nrow=4, normalize=True)
            utils.save_image(grid, os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png"))

    torch.save({
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }, "pokemon_dcgan_64_final.pth")

    print("Training abgeschlossen! Modell gespeichert als 'pokemon_dcgan_64_final.pth'")
