import numpy as np
import pandas as pd
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy_softmax(y_pred, y_true):
    m = y_true.shape[0]
    grad = softmax(y_pred)
    grad[range(m), y_true] -= 1
    grad = grad / m
    return grad

class SimpleFullyConnectedNN:
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, y_pred):
        m = y.shape[0]
        dz2 = delta_cross_entropy_softmax(y_pred, y)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        # Update the weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X_train, y_train, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            y_pred = self.forward(X_train)
            loss = cross_entropy_loss(y_pred, y_train)
            self.backward(X_train, y_train, y_pred)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean()


train_csv = pd.read_csv("D:/下载的文件/archive/fashion-mnist_train.csv")
test_csv = pd.read_csv("D:/下载的文件/archive/fashion-mnist_test.csv")
# Assuming X_train, y_train, X_test, y_test are already loaded and preprocessed
model = SimpleFullyConnectedNN()
model.train(X_train, y_train, num_epochs=10, learning_rate=0.1)
print('Training accuracy:', model.accuracy(X_train, y_train))
print('Testing accuracy:', model.accuracy(X_test, y_test))
