import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

from Digitv2Model import Digitv2Model

INPUT_SIZE: int = 28 * 28
CLASS_CNT: int = 10
TRAIN_BATCH_SZ: int = 128
TEST_BATCH_SZ: int = 1000
EPOCHS: int = 50
LEARN_RATE = 0.01


def train():
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=LEARN_RATE)

    n_steps = len(train_load)

    for epoch in range(EPOCHS):
        for i, (img, label) in enumerate(train_load):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            # forward pass
            output = model(img)
            loss: torch.Tensor = criterion(output, label).to(device)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % TRAIN_BATCH_SZ == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, step {i + 1}/{n_steps}, loss = {loss.item():.4f}")

if __name__ == "__main__":
    torch.manual_seed(0x10f2c)

    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    training_set: datasets.MNIST = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transformation
    )

    test_set: datasets.MNIST = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transformation
    )

    train_load: DataLoader = DataLoader(
        dataset=training_set,
        batch_size=TRAIN_BATCH_SZ,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    test_load: DataLoader = DataLoader(
        dataset=test_set,
        batch_size=TEST_BATCH_SZ,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = Digitv2Model().to(device)
    print(model)

    train()
