import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_dataloaders(batch_size=64):
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root=data_path,
        download=True,
        train=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        download=True,
        train=False,
        transform=transform
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