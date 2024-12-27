#torch.autograd 에 대한 간단한 소개 1
import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # 순전파 단계(forward pass)

loss = (prediction - labels).sum()
loss.backward() # 역전파 단계(backward pass)

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step() # 경사하강법(gradient descent)
