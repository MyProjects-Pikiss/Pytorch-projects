#Autograd에서 미분(differentiation) 2
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# 수집된 변화도가 올바른지 확인합니다.
print(9*a**2 == a.grad)
print(-2*b == b.grad)

#DAG에서 제외하기 3
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 신경망의 모든 매개변수를 고정합니다
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)

# 분류기만 최적화합니다.
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)