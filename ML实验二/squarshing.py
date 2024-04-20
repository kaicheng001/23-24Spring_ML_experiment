import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # 从输入层到隐藏层
        self.fc2 = nn.Linear(256, 10)   # 从隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU激活函数
        x = self.fc2(x)
        return x

def accuracy(y_hat, y):
    preds = torch.argmax(y_hat, dim=1)
    correct = (preds == y).float().sum()
    return correct / len(y)

def train(model, train_loader, test_loader, num_epochs, learning_rate):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        for X, y in train_loader:
            X = X.view(-1, 784)  # 将图片展平
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in test_loader:
                X = X.view(-1, 784)  # 将图片展平
                y_hat = model(X)
                correct += (torch.argmax(y_hat, dim=1) == y).float().sum().item()
                total += y.size(0)
            print(f'Epoch {epoch+1}, Accuracy: {correct / total:.4f}')

# 准备数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 实例化模型并训练
model = SimpleNN()
train(model, train_loader, test_loader, num_epochs=10, learning_rate=0.1)

