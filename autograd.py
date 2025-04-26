import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)

y = x ** 2 + 2 * x + 1
z= y.sum()

z.backward()
print(x.grad)


