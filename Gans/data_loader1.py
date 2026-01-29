from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(batch_size=32, image_size=128):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.95,1.05)),
        transforms.ColorJitter(0.1,0.1,0.1,0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    dataset = datasets.ImageFolder(root="data/Generate_new_Pokemon", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader
