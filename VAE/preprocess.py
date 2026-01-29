import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

DATA_DIR = "./Generate new Pokemon (1)/Generate new Pokemon/images"

# Dataset for unlabeled images 
class PokemonDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [
            f for f in os.listdir(root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

#  Transforms 
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),   # flip half the images
    transforms.RandomRotation(15),       # rotate 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

#This hopefully dramatically increase dataset but size still doesnt work properly tho
#saving image not important VAE sees differently every epoch

#  Load dataset 
full_dataset = PokemonDataset(DATA_DIR, transform=transform)

#  Split dataset 
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

#  Dataloaders 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


