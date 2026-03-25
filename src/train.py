import torch
import torch.nn as nn
from src.model import SimpleCNN
from src.data import get_dataloaders

def main():
    model = SimpleCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataloader, test_dataloader = get_dataloaders()

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(20):
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

        correct_test = 0
        total_test = 0
        correct_train = 0
        total_train = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

            for images, labels in train_dataloader:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

        test_accuracy = correct_test / total_test
        train_accuracy = correct_train / total_train

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Test Acc: {test_accuracy:.4f}, Train Acc: {train_accuracy:.4f}")


if __name__ == "__main__":
    main()
