import torch
import torch.nn as nn

# MSE
mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad = True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)

# Categorical Cross-Entropy Loss
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
print(loss)

# Binary Cross-Entropy Loss
bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
loss = bce_loss(probabilities, targets)
print(probabilities)
print(loss)

# Adam optimizer
import torch.optim as optim

input_dim = 2
lr = 0.001

perceptron = Perceptron(input_dim=input_dim)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(parpreprocess_textams=perceptron.parameters(), lr=lr)