import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

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


train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
    )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
    )