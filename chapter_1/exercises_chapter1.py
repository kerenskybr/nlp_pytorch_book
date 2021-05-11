import torch

# 1. Create a 2D tensor and then add a dimension of size 1 inserted at dimension 0.
a = torch.rand(3, 3)
print(a.unsqueeze(0))

# 2. Remove the extra dimension you just added to the previous tensor.
print(a.squeeze(0))

# 3. Create a random tensor of shape 5x3 in the interval [3, 7)
print(3 + torch.rand(5, 3) * (7 - 3))

# 4. Create a tensor with values from a normal distribution (mean=0, std=1).
a = torch.rand(3, 3)
print(a)

# 5. Retrieve the indexes of all the nonzero elements in the tensor torch.Tensor([1, 1, 1,0, 1]).. Create a random tensor of size (3,1) and then horizontally stack four copies together.
a = torch.Tensor([1, 1, 1, 0, 1])
print(torch.nonzero(a))

# 7. Return the batch matrix­matrix product of two three­dimensional matrices(a=torch.rand(3,4,5), b=torch.rand(3,5,4)).
a = torch.rand(3, 4, 5)
b = torch.rand(3, 5, 4)
print(torch.bmm(a, b))
# 8. Return the batch matrix­matrix product of a 3D matrix and a 2D matrix(a=torch.rand(3,4,5), b=torch.rand(5,4)).
a = torch.rand(3, 4, 5)
b = torch.rand(5, 4)
print(torch.bmm(a, b.unsqueeze(0).expand(a.size(0), *b.size())))