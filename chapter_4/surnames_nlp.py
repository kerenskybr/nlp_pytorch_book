import torch
from torch.nn import Conv1d

batch_size = 2
one_hot_size = 10
sequence_width = 7
data = torch.randn(batch_size, one_hot_size, sequence_width)
conv1 = Conv1d(in_channels=one_hot_size, out_channels=16, kernel_size=3)
intermediate1 = conv1(data)
print(data.size())
print(intermediate1.size())

conv2 = Conv1d(in_channels=16, out_channels=32, kernel_size=3)
conv3 = Conv1d(in_channels=32, out_channels=64, kernel_size=3)

intermediate2 = conv2(intermediate1)
intermediate3 = conv3(intermediate2)
print("*"*20)
print(intermediate2.size())
print(intermediate3.size())

y_output = intermediate3.squeeze()
print(y_output.size())

# Method 2 of reducing to feature vectors
print(intermediate1.view(batch_size, -1).size())
# Method 3 of reducing to feature vectors
print(torch.mean(intermediate1, dim=2).size())
# print(torch.max(intermediate1, dim=2).size())
# print(torch.sum(intermediate1, dim=2).size())