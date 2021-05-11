import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Sigmoid example
x = torch.arange(-5.0, 5.0, 0.1)
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.numpy())
plt.xlabel("sigmoid")
plt.show()

# tahn example
x = torch.arange(-5.0, 5.0, 0.1)
y = torch.tanh(x)
plt.plot(x.numpy(), y.numpy())
plt.xlabel("tahn")
plt.show()

# Relu example
relu = torch.nn.ReLU()
x = torch.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x.numpy(), y.numpy())
plt.xlabel("relu")
plt.show()

# PReRelu example
prelu = torch.nn.PReLU()
x = torch.arange(-5.0, 5.0, 0.1)
y = prelu(x)
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.xlabel("prelu")
plt.show()

# Softmax example
softmax = nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_output = softmax(x_input)
print(x_input)
print(y_output)
print(torch.sum(y_output, dim=1))