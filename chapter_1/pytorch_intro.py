import torch
import numpy as np

def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


describe(torch.Tensor(2, 3))
describe(torch.rand(2, 3))
describe(torch.randn(2, 3))
print('-'*10)
describe(torch.zeros(2, 3))
x = torch.ones(2, 3)
describe(x)
x.fill_(5)
describe(x)


# Creating and initializing a tensor from NumPy
npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))

x = torch.FloatTensor([[1, 2, 3],[4, 5, 6]])
describe(x)