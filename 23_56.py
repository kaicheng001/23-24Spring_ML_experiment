import numpy as np
import matplotlib as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 加载数据集


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleClassifier:
  # ReLU函数
 def relu(Z):
    return np.maximum(0, Z)

# ReLU函数的导数
 def d_relu(Z):
    return Z > 0

# Softmax函数
 def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

# 初始化模型参数
 def initialize_parameters(input_size, hidden_size, output_size):
    parameters = {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.randn(hidden_size, output_size) * 0.01,
        "b2": np.zeros((1, output_size))
    }
    return parameters

# 前向传播
 def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    Z1 = np.dot(X, W1) + b1
    A1 = SimpleClassifier.relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = SimpleClassifier.softmax(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# 交叉熵损失
 def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = -np.mean(Y * np.log(A2 + 1e-9))
    return cost

# 反向传播
 def backward_propagation(X, Y, cache, parameters):
    Z1, A1, Z2, A2 = cache
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    m = X.shape[0]
    
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * SimpleClassifierd_relu(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# 更新参数
 def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters



import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 您已经有的 SimpleClassifier 类代码

def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    parameters = model.initialize_parameters(input_size, hidden_size, output_size)
    train_costs, test_costs = [], []
    
    for epoch in range(num_epochs):
        model.train() # 设置模型到训练模式
        for images, labels in train_loader:
            images = images.reshape(-1, 28*28).numpy() # Flatten the images
            labels = one_hot_encode(labels.numpy()) # One hot encode labels
            outputs, caches = model.forward_propagation(images, parameters)
            cost = model.compute_cost(outputs, labels)
            grads = model.backward_propagation(images, labels, caches, parameters)
            parameters = model.update_parameters(parameters, grads, learning_rate)
        
        # 记录训练损失
        train_costs.append(cost)

        # 在测试集上评估模型
        model.eval() # 设置模型到评估模式
        with torch.no_grad(): # 禁用梯度计算
            test_cost = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).numpy()
                labels = one_hot_encode(labels.numpy())
                outputs, _ = model.forward_propagation(images, parameters)
                test_cost = model.compute_cost(outputs, labels)
            test_costs.append(test_cost)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {cost}, Test Loss: {test_cost}')

    # 可视化训练集与验证集上的损失
    plt.plot(train_costs, label='Training Loss')
    plt.plot(test_costs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def one_hot_encode(labels, num_classes=10):
    # 将标签转化为 one-hot 编码
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# 设置训练参数
input_size = 28 * 28 # 图片尺寸 28x28
hidden_size = 128
output_size = 10
num_epochs = 10
learning_rate = 0.1

# 实例化模型
model = SimpleClassifier()
# 获取数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# 开始训练模型
train_model(model, train_loader, test_loader, num_epochs, learning_rate)
