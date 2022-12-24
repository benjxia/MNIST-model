import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=28,
                           hidden_size=128,
                           num_layers=2,
                           batch_first=True)
        self.lin1 = nn.Linear(128, 10)


    def forward(self, x):
        x = x.reshape(-1, 28, 28).to(device)

        out, _ = self.rnn1(x)
        out = out[:, -1, :]
        out = self.lin1(out)
        return out

train_set = torchvision.datasets.MNIST(
    root="../../data",
    transform=transforms.ToTensor(),
    download=True,
    train=True
)

test_set = torchvision.datasets.MNIST(
    root="../../data",
    transform=transforms.ToTensor(),
    download=True,
    train=False
)
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=False)

model = RNN_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:

        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
