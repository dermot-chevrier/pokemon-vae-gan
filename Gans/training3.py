import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from model import Generator, Discriminator
from data_loader import get_dataloader

EPOCHS = 200
BATCH_SIZE = 16
IMAGE_SIZE = 64
LATENT_DIM = 64

LR = 1e-4
BETAS = (0.0, 0.9)
N_CRITIC = 5
LAMBDA_GP = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("gen_images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

G = Generator(z_dim=LATENT_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)

optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

dataloader = get_dataloader(
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

def gradient_penalty(D, real, fake):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=DEVICE)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    d_interpolated = D(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)

print("🚀 Training gestartet")

for epoch in range(1, EPOCHS + 1):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)

        for _ in range(N_CRITIC):
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = G(noise)

            D_real = D(real_images).mean()
            D_fake = D(fake_images.detach()).mean()

            gp = gradient_penalty(D, real_images, fake_images)

            loss_D = -(D_real - D_fake) + LAMBDA_GP * gp

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
        fake_images = G(noise)
        loss_G = -D(fake_images).mean()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"| D Loss: {loss_D.item():.4f} "
        f"| G Loss: {loss_G.item():.4f}"
    )

    with torch.no_grad():
        fake = G(fixed_noise)
        save_image(
            fake,
            f"gen_images/epoch_{epoch:03d}.png",
            normalize=True,
            nrow=4
        )

torch.save(G.state_dict(), "checkpoints/generator.pth")
torch.save(D.state_dict(), "checkpoints/discriminator.pth")

print("✅ Training abgeschlossen & Modelle gespeichert")
