import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt
# 假设配置文件已经定义，并包含了一些基本设置
import ML实验二.pytorch_fashion.config
args = ML实验二.pytorch_fashion.config.args
device = torch.device('cpu' if args.cpu else 'cuda')

d2l.use_svg_display()

# Animator类定义
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', fmts=None, nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        if fmts is None:
            fmts = ['-', 'm--', 'g-.', 'r:']
        plt.ion()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# 自定义数据集类
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        # data是从torchvision.datasets中获取的，包含data和targets
        self.point = data.data.type(torch.FloatTensor)  # 将数据转换为浮点张量
        self.label = data.targets.type(torch.FloatTensor)  # 将标签转换为浮点张量
        self.transform = transform

    def __getitem__(self, index):
        x = self.point[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.label)

# 数据变换
trans = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化处理
])

# 加载原始FashionMNIST数据集
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

# 使用自定义数据集类创建数据集
train_data = my_dataset(mnist_train)
test_data = my_dataset(mnist_test)

# 创建数据加载器
train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, input_size=784, hidden_layer_size=4, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)  # 展平图像张量为向量
        x = F.relu(self.fc1(x))  # 第一层加ReLU激活
        x = self.fc2(x)  # 输出层
        return x

# 实例化模型并设置到适当的设备上
model = Classifier()
model.to(device)
