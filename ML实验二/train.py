import config
import torch
args=config.args
device=torch.device('cpu' if args.cpu else 'cuda')

#4.	读入数据需要重写torch中的dataset类
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        # pytorh的训练集必须是tensor形式，可以直接在dataset类中转换，省去了定义transform
        # 转换Y数据类型为长整型
        self.point = torch.from_numpy(x).type(torch.FloatTensor)
        self.label = torch.from_numpy(y).type(torch.FloatTensor)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.point[index]
        label = self.label[index]
        # print(x)
        return x, label

    def __len__(self):
        return len(self.label)


train_data = my_dataset(X, Y)
train_loader = data.DataLoader(train_data, batch_size=args.bacth_size, shuffle=True)

#5.模型需要继承nn.Module
class Classifier(nn.Module):
    # 初始化函数，对网路的输入层、隐含层、输出层的大小和使用的函数进行了规定。
    def __init__(self, input_size=2, hidden_layersize=4):
        super(Classifier, self).__init__()
       	#yourconde
        
    def forward(self, x):
        
#your code
        
        return x

