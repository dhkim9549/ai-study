import torch
import math

x = torch.tensor([2., 1.], requires_grad=True)
print(f'x = {x}')

y = x * 3 
print(f'y = {y}')

z = y[0] ** 2 + y[1] ** 2 
print(f'z = {z}')

z.backward()
print(f'x.grad = {x.grad}')
