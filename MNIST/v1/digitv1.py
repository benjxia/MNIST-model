import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE: int = 28 * 28
OUTPUT_CLS: int = 10
TRAIN_BATCH_SZ: int = 64
TEST_BATCH_SZ: int = 1000
TRAIN_EPOCH: int = 10
LEARNING_RATE: float = 0.01
MOMENTUM: float = 0.5
LOG_INTERVAL: int = 10

RANDOM_SEED: int = 0x10f2c
torch.manual_seed(RANDOM_SEED)

training_set: datasets.MNIST = datasets.MNIST(
    root="../../data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_set: datasets.MNIST = datasets.MNIST(
    root="../../data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dl: DataLoader = DataLoader(training_set, batch_size=TRAIN_BATCH_SZ, shuffle=True)
test_dl: DataLoader = DataLoader(test_set, batch_size=TEST_BATCH_SZ, shuffle=False)

device: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DigitNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(INPUT_SIZE, 100)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def train():
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_steps = len(train_dl)

    for epoch in range(TRAIN_EPOCH):
        for i, (img, label) in enumerate(train_dl):
            img = img.to(device)
            label = label.to(device)
            img = img.reshape(-1, 28 * 28)

            # forward pass
            output = model(img)
            loss: torch.Tensor = criterion(output, label)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % TRAIN_BATCH_SZ == 0:
                print(f"Epoch {epoch + 1}/{TRAIN_EPOCH}, step {i + 1}/{n_steps}, loss = {loss.item():.4f}")


def test():
    with torch.no_grad():
        correct = 0
        samples = 0
        for img, labels in test_dl:
            img = img.to(device)
            labels = labels.to(device)
            img = img.reshape(-1, 28 * 28)
            output = model(img)

            _, predictions = torch.max(output, 1)
            samples += labels.shape[0]
            correct += (predictions == labels).sum().item()

    acc = 100 * correct / samples
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    model = DigitNetwork().to(device)
    print(model)
    train()

