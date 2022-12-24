import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

from Digitv2Model import Digitv2Model

INPUT_SIZE: int = 28 * 28
CLASS_CNT: int = 10
TRAIN_BATCH_SZ: int = 128
TEST_BATCH_SZ: int = 1000
EPOCHS: int = 50
LEARN_RATE = 0.01


if __name__ == "__main__":

    torch.manual_seed(0x10f2c)

    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    training_set: datasets.MNIST = datasets.MNIST(
        root= "../../data",
        train= True,
        download= True,
        transform= transformation
    )

    test_set: datasets.MNIST = datasets.MNIST(
        root= "../../data",
        train= False,
        download= True,
        transform= transformation
    )

    train_load: DataLoader = DataLoader(
        dataset= training_set,
        batch_size= TRAIN_BATCH_SZ,
        shuffle= True,
        pin_memory= True,
        num_workers= 4
    )

    test_load: DataLoader = DataLoader(
        dataset= test_set,
        batch_size= TEST_BATCH_SZ,
        shuffle= False,
        pin_memory= True,
        num_workers= 4
    )

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = Digitv2Model().to(device)
    print(model)

    model.load_state_dict(torch.load("./digitsv2.pth", map_location=torch.device("cpu")))
    model.eval()


    with torch.no_grad():
        correct = 0
        total = 0
        for img, labels in test_load:
            img = img.to(device, non_blocking= True)
            labels = labels.to(device, non_blocking= True)

            output = model(img)
            _, prediction = torch.max(output, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        acc = 100.0 * correct / total
        print(f"Accuracy: {acc}%")
        # 98.49% accurate MNIST model
