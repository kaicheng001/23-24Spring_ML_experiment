# predict.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class CustomNetwork(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(CustomNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义输出标签的函数
def output_label(label):
    output_mapping = {
             0: "T-shirt/Top",
             1: "Trouser",
             2: "Pullover",
             3: "Dress",
             4: "Coat", 
             5: "Sandal", 
             6: "Shirt",
             7: "Sneaker",
             8: "Bag",
             9: "Ankle Boot"
    }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

def load_model(path):
    model = CustomNetwork(784, 256, 10)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict(model, dataloader):
    for images, labels in dataloader:
        images = images.view(-1, 784)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # 转换预测结果的索引到类别的名称
        predicted_labels = [output_label(prediction) for prediction in predictions]
        return predicted_labels  # 返回类别的名称

def main():
    # 加载测试数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 加载模型
    model = load_model('fashion_mnist.pt')

    # 进行预测
    predictions = predict(model, test_loader)
    print(predictions)

if __name__ == '__main__':
    main()
