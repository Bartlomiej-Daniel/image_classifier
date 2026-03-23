import torch
import torch.nn as nn
from src.model import SimpleCNN
from src.data import get_dataloaders

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader, test_dataloader = get_dataloaders()

for epoch in range(10):
    model.train()
    total_loss = 0

    for images, labels in train_dataloader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.4f}")


