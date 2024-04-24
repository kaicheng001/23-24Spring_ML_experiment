import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, num_inputs=784, num_hiddens=256, num_outputs=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hiddens)  # Input layer to hidden layer
        self.fc2 = nn.Linear(num_hiddens, num_outputs)  # Hidden layer to output layer
        self.relu = nn.ReLU()  # ReLU activation function

        # Initialize weights and biases
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Pass the input through the first layer, then apply ReLU
        x = self.fc2(x)  # Pass through the second layer
        return x

# Create an instance of Classifier
model = Classifier()


