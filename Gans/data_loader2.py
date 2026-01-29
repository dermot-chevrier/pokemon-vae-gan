import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Neuer Pfad: nur der Unterordner mit Bildern
DATA_DIR = "data/Generate_new_Pokemon/Generate_new_Pokemon"

def get_dataloader(batch_size=16, image_size=64, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    # ImageFolder erwartet übergeordneten Ordner, deshalb dummy-class 'pokemon'
    dataset = datasets.ImageFolder(
        root=os.path.dirname(DATA_DIR),  # übergeordneter Ordner
        transform=transform
    )

    # Filter nur den Unterordner Generate_newPokemon
    dataset.samples = [s for s in dataset.samples if "Generate_new_Pokemon" in s[0]]
    dataset.targets = [0]*len(dataset.samples)  # Labels egal für GAN

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    return dataloader

# Test
if __name__ == "__main__":
    loader = get_dataloader()
    images, _ = next(iter(loader))
    print("Batch shape:", images.shape)
    print("DataLoader Length:", len(loader))
