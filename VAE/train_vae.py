import torch
from torch import optim
from torchvision.utils import save_image
import os
import time
import matplotlib.pyplot as plt
from model import VAE, vae_loss_function
from preprocess import train_loader, test_loader
from tqdm import tqdm

if torch.cuda.is_available():
    print(f" Using GPU")
else:
    print("shit aint work ahgain")

for x in train_loader:
    print(x.shape)
    break

# Hyperparameter

LATENT_DIM = 64
EPOCHS = 250
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# Initializeing model

vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

# Track time and losses
start_time = time.time()
losses = []
val_losses = []



# Training loop

print(" Starting training...")
for epoch in range(1, EPOCHS + 1):
    vae.train()
    epoch_loss = 0

    for batch_idx, x in enumerate(tqdm(train_loader)):

        x = x.to(DEVICE)
        optimizer.zero_grad()

        recon_x, mu, logvar = vae(x)
        loss = vae_loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader.dataset)
    losses.append(avg_loss)
    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {avg_loss:.4f}")

    
    # Validation loss
    vae.eval()
    val_loss = 0
    for x_val in test_loader:
        x_val = x_val.to(DEVICE)
        recon_x, mu, logvar = vae(x_val)
        loss = vae_loss_function(recon_x, x_val, mu, logvar)
        val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Save sample reconstruction
    vae.eval()
    with torch.no_grad():
        val_sample = next(iter(test_loader))[:16].to(DEVICE)   # take first 16 images
        recon, _, _ = vae(val_sample)
        save_image(recon, os.path.join("keep", f"recon_val_epoch{epoch}.png"))


    # Saveing checkpoints
    if (epoch + 1) % 10 == 0:
        torch.save(vae.state_dict(), f"checkpoints/vae_epoch{epoch+1}.pth")


# Final saveeee plus time
torch.save(vae.state_dict(), "checkpoints/vae_final.pth")
duration = time.time() - start_time
print(f" Training complete in {duration:.2f} seconds.")

# plottingg
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training vs. Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/loss_comparison_plot.png")
plt.show()