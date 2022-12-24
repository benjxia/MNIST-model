import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

INPUT_SIZE: int = 28 * 28
CLASS_CNT: int = 10
TRAIN_BATCH_SZ: int = 100
TEST_BATCH_SZ: int = 1000
EPOCHS: int = 100
LEARN_RATE = 0.001

torch.manual_seed(42)

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


class Digitv3Network(nn.Module):
    def __init__(self):
        super(Digitv3Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 7, bias=False)  # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)  # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False)  # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, 10, bias=False)  # 192 * 16 = 3072
        self.fc1_bn = nn.BatchNorm1d(10)

    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits

    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)


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


def test():
    with torch.no_grad():
        correct = 0
        total = 0
        for img, labels in test_load:
            img = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(img)
            _, prediction = torch.max(output, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        acc = 100.0 * correct / total
        print(f"Accuracy: {acc}%")


if __name__ == "__main__":
    model = Digitv3Network().to(device)
    model.load_state_dict(torch.load("./digitsv3.pth"))
    model.eval()
    print(model)
