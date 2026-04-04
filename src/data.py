import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_dataloaders(batch_size=64, img_size=32):
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data"

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root=data_path,
        download=True,
        train=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        download=True,
        train=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader